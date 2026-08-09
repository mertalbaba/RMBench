from ._base_task import Base_Task
from .utils import *
from sapien.render import set_ray_tracing_denoiser
import json
import os

# ---------------------------------------------------------------------------
# long_horizon: scripted multi-stage episodes for memory-stream supervision.
#
# An episode is a seeded SCRIPT of 6+ stages sampled from a stage library:
#   relocate     move a block to the (single) free mat
#   cover        hide a target block under one of two identical covers
#   uncover      return the cover hiding a NAMED block to its park spot --
#                with both covers down this is a shell-game probe: the covers
#                are visually identical, so acting on "Uncover the red block."
#                requires remembering which cover went over which block
#   return_home  move a block back to the mat it started the episode on
#                (memory over the whole horizon, put_back_block style)
#   press        press the button k times (distractor / confirm)
#
# The script is generated in load_actors from np.random (seeded by the base
# class), so seed -> scene + script is fully deterministic and replays
# identically during data collection. Stage count is the scaling knob:
# 6-10 stages ~= 1200-2200 recorded frames; 40+ stages ~= 10k frames.
#
# Scene templates 0/1 are train, template 2 is eval (held-out layout AND
# held-out block colors purple/orange). Template id + object identities are
# tagged into scene_info.json (key "long_horizon") for split control.
#
# Ground-truth events + stage timings are written per episode to
# data/episode{N}_event_log.json next to the HDF5 (frame-index timestamps at
# the same _take_picture call sites that produce language_annotation.json).
# ---------------------------------------------------------------------------

_COLOR_RGB = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "purple": (0.5, 0.0, 0.8),
    "orange": (1.0, 0.5, 0.0),
}

_TEMPLATES = {
    # NOTE: keep all manipulated xy inside the validated reach envelope:
    # x in [0.0, 0.30], y in [-0.22, -0.03] for the right arm. The original
    # template 0 had its back mat at y=+0.03 and every script touching it
    # failed to plan (grasp, place and cover all fail there).
    0: {
        "mats": {"left": (0.00, -0.12), "right": (0.20, -0.12), "front": (0.10, -0.22), "back": (0.10, -0.03)},
        "parks": [(0.30, -0.22), (0.30, -0.03)],
        "button": (-0.25, -0.10),
        "colors": ("red", "green", "blue"),
        "mat_rgb": (0.65, 0.50, 0.30),
    },
    1: {
        "mats": {"left": (0.00, -0.14), "right": (0.20, -0.14), "front": (0.10, -0.22), "back": (0.10, -0.04)},
        "parks": [(0.30, -0.22), (0.30, -0.04)],
        "button": (-0.20, -0.15),
        "colors": ("yellow", "blue", "red"),
        "mat_rgb": (0.25, 0.25, 0.25),
    },
    2: {
        "mats": {"left": (0.00, -0.13), "right": (0.22, -0.13), "front": (0.11, -0.22), "back": (0.11, -0.04)},
        "parks": [(0.32, -0.22), (0.32, -0.04)],
        "button": (-0.18, -0.08),
        "colors": ("purple", "orange", "green"),
        "mat_rgb": (0.90, 0.90, 0.90),
    },
}

_TRAIN_TEMPLATES = (0, 1)
_EVAL_TEMPLATES = (2,)


class long_horizon(Base_Task):

    def setup_demo(self, **kwags):
        # split hook: task_config keys override, else eval_mode selects the
        # held-out template + colors automatically
        self._lh_split = kwags.get("lh_split") or ("eval" if kwags.get("eval_mode", False) else "train")
        default_templates = _EVAL_TEMPLATES if self._lh_split == "eval" else _TRAIN_TEMPLATES
        self._lh_templates = tuple(kwags.get("lh_templates") or default_templates)
        # default range targets the v0 budget of 1500-3000 recorded frames
        # (~141 frames/stage measured); smoke configs override it downward
        self._lh_stage_range = tuple(kwags.get("lh_stage_range") or (11, 21))
        self._lh_denoiser_off = bool(kwags.get("lh_disable_denoiser", False))
        self._lh_seed = int(kwags.get("seed", 0))
        super()._init_task_env_(**kwags)

    def setup_scene(self, **kwargs):
        super().setup_scene(**kwargs)
        if getattr(self, "_lh_denoiser_off", False):
            # OIDN dies under GPU contention ("OIDN Error: invalid handle"
            # spam; frames keep RT grain but stay valid). Turning it off is
            # config-scoped to this task and only changes denoising.
            set_ray_tracing_denoiser("none")

    # ------------------------------------------------------------ scene setup
    def load_actors(self):
        idx = int(np.random.randint(len(self._lh_templates)))
        self.template_id = int(self._lh_templates[idx])
        cfg = _TEMPLATES[self.template_id]

        self.mat_names = list(cfg["mats"].keys())
        self.mats, self.mat_xy = {}, {}
        for name, (x, y) in cfg["mats"].items():
            self.mats[name] = create_box(
                scene=self,
                pose=rand_pose(xlim=[x, x], ylim=[y, y], qpos=[1, 0, 0, 0]),
                half_size=(0.04, 0.04, 0.0005),
                color=cfg["mat_rgb"],
                name=f"{name}_mat",
                is_static=True,
            )
            self.mat_xy[name] = (float(x), float(y))

        colors = list(cfg["colors"])
        order = [int(i) for i in np.random.permutation(3)]
        self.target1_color = colors[order[0]]
        self.target2_color = colors[order[1]]
        self.distractor_color = colors[order[2]]

        start_mats = [self.mat_names[int(i)] for i in np.random.permutation(len(self.mat_names))[:3]]
        self.blocks, self.block_home = {}, {}
        for color, mat_name in zip(
                (self.target1_color, self.target2_color, self.distractor_color), start_mats):
            x, y = self.mat_xy[mat_name]
            self.blocks[color] = create_box(
                scene=self,
                pose=rand_pose(xlim=[x, x], ylim=[y, y], zlim=[0.741 + 0.02], qpos=[1, 0, 0, 0]),
                half_size=(0.02, 0.02, 0.02),
                color=_COLOR_RGB[color],
                name=f"{color}_block",
            )
            self.block_home[color] = mat_name

        self.park_xy = [(float(x), float(y)) for x, y in cfg["parks"]]
        self.covers = []
        for x, y in self.park_xy:
            self.covers.append(
                create_actor(
                    scene=self,
                    pose=rand_pose(xlim=[x, x], ylim=[y, y], qpos=[0.5, 0.5, 0.5, 0.5]),
                    modelname="003_cover",
                    model_id=0,
                    convex=True,
                ))
        self.quat_of_target_pose = [0.0, 1.0, 0.0, 0.0]

        bx, by = cfg["button"]
        self.button_xy = (float(bx), float(by))
        self.button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[bx, bx],
            ylim=[by, by],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button.set_mass(0.0001, ["button_cap"])
        self.set_button_unpressed(self.button)
        self.press_cnt = 0
        self.press_flag = False

        self.script, self.expected_final = self._generate_script()
        print(f"[long_horizon] seed {self._lh_seed} template {self.template_id} "
              f"stages: {[s['type'] for s in self.script]}", flush=True)
        self.n_stages = len(self.script)
        self.expected_press_total = int(self.expected_final["press_total"])
        self._tracker_ptr = 0
        self._script_fail = False
        self.stage_records = []

    # --------------------------------------------------------- script sampler
    def _stage_annotation(self, stage):
        typ, c = stage["type"], stage.get("block")
        if typ == "relocate":
            return f"Move the {c} block to the {stage['to']} mat."
        if typ == "return_home":
            return f"Move the {c} block back to its original mat."
        if typ == "cover":
            return f"Cover the {c} block with a cover."
        if typ == "uncover":
            return f"Uncover the {c} block."
        if typ == "press":
            return "Press the button one time."
        raise ValueError(typ)

    def _generate_script(self):
        lo, hi = self._lh_stage_range
        n_target = int(np.random.randint(lo, hi + 1))

        t1, t2, d = self.target1_color, self.target2_color, self.distractor_color
        loc = dict(self.block_home)  # color -> mat name (symbolic state)
        covered = {}                 # color -> cover idx
        press_total = 0
        stages = []

        def free_mat():
            used = set(loc.values())
            return [m for m in self.mat_names if m not in used][0]

        def emit(typ, **kw):
            nonlocal press_total
            stage = {"type": typ, **kw}
            if typ in ("relocate", "return_home"):
                b = stage["block"]
                stage["from"] = loc[b]
                loc[b] = stage["to"]
            elif typ == "cover":
                covered[stage["block"]] = stage["cover"]
                stage["mat"] = loc[stage["block"]]
            elif typ == "uncover":
                stage["cover"] = covered.pop(stage["block"])
                stage["mat"] = loc[stage["block"]]
            elif typ == "press":
                press_total += stage["k"]
            stage["press_cum"] = press_total
            stage["annotation"] = self._stage_annotation(stage)
            stages.append(stage)

        def emit_filler():
            fm = free_mat()
            options = [("relocate", b) for b in (d, t2, t1)
                       if b not in covered and loc[b] != fm]
            if press_total < 3 and (not stages or stages[-1]["type"] != "press"):
                options.append(("press", None))
            typ, b = options[int(np.random.randint(len(options)))]
            if typ == "press":
                emit("press", k=int(np.random.randint(1, 3)))
            else:
                emit("relocate", block=b, to=fm)

        # motif: establish -> occlude -> distract -> (occlude 2nd) -> probe
        emit("relocate", block=t1, to=free_mat())
        emit("cover", block=t1, cover=0)
        emit_filler()
        both = n_target >= 8 or (n_target >= 7 and np.random.rand() < 0.5)
        if both:
            emit("cover", block=t2, cover=1)
            emit_filler()

        tail_len = (2 if both else 1) + 2  # uncovers + final move + final press
        while len(stages) + tail_len < n_target:
            emit_filler()

        if both:
            first = t1 if np.random.rand() < 0.5 else t2
            second = t2 if first == t1 else t1
            emit("uncover", block=first)
            emit("uncover", block=second)
        else:
            emit("uncover", block=t1)

        # final move: prefer return_home (long-horizon memory), else filler
        fm = free_mat()
        home_candidates = [b for b in (t1, t2, d)
                           if b not in covered and loc[b] != self.block_home[b] and self.block_home[b] == fm]
        if home_candidates:
            b = home_candidates[int(np.random.randint(len(home_candidates)))]
            emit("return_home", block=b, to=self.block_home[b])
        else:
            movable = [b for b in (t1, t2, d) if b not in covered]
            emit("relocate", block=movable[int(np.random.randint(len(movable)))], to=fm)

        if stages[-1]["type"] != "press":
            emit("press", k=1)

        expected_final = {
            "block_loc": dict(loc),
            "covers_parked": True,
            "press_total": int(press_total),
        }
        return stages, expected_final

    # ------------------------------------------------------- geometry checks
    def _on_mat(self, block, mat_name):
        p = block.get_pose().p
        mx, my = self.mat_xy[mat_name]
        return np.abs(p[0] - mx) < 0.035 and np.abs(p[1] - my) < 0.035 and p[2] < 0.78

    def _cover_on_mat(self, cover_idx, mat_name):
        p = self.covers[cover_idx].get_pose().p
        mx, my = self.mat_xy[mat_name]
        return np.abs(p[0] - mx) < 0.03 and np.abs(p[1] - my) < 0.03 and p[2] < 0.744

    def _cover_parked(self, cover_idx):
        p = self.covers[cover_idx].get_pose().p
        px, py = self.park_xy[cover_idx]
        return np.abs(p[0] - px) < 0.05 and np.abs(p[1] - py) < 0.05 and p[2] < 0.75

    def _stage_done(self, stage):
        typ = stage["type"]
        if typ in ("relocate", "return_home"):
            return self._on_mat(self.blocks[stage["block"]], stage["to"])
        if typ == "cover":
            return self._cover_on_mat(stage["cover"], stage["mat"])
        if typ == "uncover":
            return self._cover_parked(stage["cover"])
        if typ == "press":
            return self.press_cnt >= stage["press_cum"]
        return False

    def _update_stage_tracker(self):
        while self._tracker_ptr < self.n_stages and self._stage_done(self.script[self._tracker_ptr]):
            self._tracker_ptr += 1

    # ------------------------------------------------------- stage execution
    def _exec_stage(self, stage):
        ann = stage["annotation"]
        typ = stage["type"]
        if typ in ("relocate", "return_home"):
            block = self.blocks[stage["block"]]
            self.move(self.grasp_actor(block, arm_tag="right", pre_grasp_dis=0.09, grasp_dis=0.02),
                      language_annotation=ann)
            self.move(self.move_by_displacement(arm_tag="right", z=0.15), language_annotation=ann)
            target = self.mats[stage["to"]].get_pose().p
            self.move(self.place_actor(block, arm_tag="right", target_pose=target,
                                       functional_point_id=2, dis=0.01), language_annotation=ann)
            self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=ann)
        elif typ == "cover":
            cover = self.covers[stage["cover"]]
            bp = self.blocks[stage["block"]].get_pose().p
            target = [float(bp[0]), float(bp[1]), 0.741] + self.quat_of_target_pose
            self.move(self.grasp_actor(cover, arm_tag="right", pre_grasp_dis=0.05), language_annotation=ann)
            self.move(self.move_by_displacement(arm_tag="right", z=0.08), language_annotation=ann)
            self.move(self.place_actor(cover, arm_tag="right", target_pose=target,
                                       functional_point_id=0, pre_dis=0.05, dis=0.005), language_annotation=ann)
            self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=ann)
        elif typ == "uncover":
            cover = self.covers[stage["cover"]]
            px, py = self.park_xy[stage["cover"]]
            target = [px, py, 0.741] + self.quat_of_target_pose
            self.move(self.grasp_actor(cover, arm_tag="right", pre_grasp_dis=0.05), language_annotation=ann)
            self.move(self.move_by_displacement(arm_tag="right", z=0.08), language_annotation=ann)
            self.move(self.place_actor(cover, arm_tag="right", target_pose=target,
                                       functional_point_id=0, pre_dis=0.05, dis=0.005), language_annotation=ann)
            self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=ann)
        elif typ == "press":
            for _ in range(stage["k"]):
                self.move(self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08,
                                           grasp_dis=0.08, contact_point_id=0), language_annotation=ann)
                self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=ann)
                self.update_press_success(self.button, "press_flag", "press_cnt")
                self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=ann)
                self.set_button_unpressed(self.button)
                self.update_button_reset(self.button, "press_flag")
            self.move(self.back_to_origin(arm_tag="left"), language_annotation=ann)

    def play_once(self):
        self.stage_records = []
        for i, stage in enumerate(self.script):
            t0 = self.FRAME_IDX
            print(f"[long_horizon] ep{self.ep_num} seed{self._lh_seed} "
                  f"stage {i + 1}/{self.n_stages} {stage['type']} start (frame {t0})", flush=True)
            self._exec_stage(stage)
            if not self.plan_success:
                print(f"[long_horizon] seed {self._lh_seed} PLAN FAILURE during stage "
                      f"{i + 1}/{self.n_stages} {stage}", flush=True)
                return self.info
            t1 = self.FRAME_IDX
            self._update_stage_tracker()
            ok = self._tracker_ptr > i
            if not ok:
                self._script_fail = True
                print(f"[long_horizon] seed {self._lh_seed} POSTCONDITION FAILED after stage "
                      f"{i + 1}/{self.n_stages} {stage}", flush=True)
            self.stage_records.append({
                "stage": i,
                "type": stage["type"],
                "annotation": stage["annotation"],
                "t0": int(t0),
                "t1": int(t1),
                "ok": bool(ok),
                "events": self._stage_events(stage, int(t1)),
            })

        self.info["info"] = {"stage_plan": self._stage_plan_text()}
        self.info["long_horizon"] = self._lh_metadata()
        if self.save_data:
            self._write_event_log()
        return self.info

    # -------------------------------------------------------- event logging
    def _stage_events(self, stage, t):
        typ = stage["type"]
        if typ in ("relocate", "return_home"):
            return [{"event": "block_moved", "block": stage["block"],
                     "from": stage["from"], "to": stage["to"], "t": t}]
        if typ == "cover":
            return [{"event": "block_hidden", "block": stage["block"],
                     "cover": stage["cover"], "mat": stage["mat"], "t": t}]
        if typ == "uncover":
            return [{"event": "block_revealed", "block": stage["block"],
                     "cover": stage["cover"], "mat": stage["mat"], "t": t}]
        if typ == "press":
            return [{"event": "button_pressed", "presses": stage["k"],
                     "count_after": stage["press_cum"], "t": t}]
        return []

    def _stage_plan_text(self):
        parts = []
        for i, stage in enumerate(self.script):
            ann = stage["annotation"]
            body = ann[0].lower() + ann[1:]
            head = "First" if i == 0 else ("Finally" if i == len(self.script) - 1 else "Then")
            parts.append(f"{head}, {body}")
        return " ".join(parts)

    def _lh_metadata(self):
        return {
            "template_id": int(self.template_id),
            "split": self._lh_split,
            "seed": int(self._lh_seed),
            "colors": {
                "target1": self.target1_color,
                "target2": self.target2_color,
                "distractor": self.distractor_color,
            },
            "block_home": dict(self.block_home),
            "mats": {k: list(v) for k, v in self.mat_xy.items()},
            "parks": [list(p) for p in self.park_xy],
            "button_xy": list(self.button_xy),
            "n_stages": int(self.n_stages),
            "stages": self.script,
            "expected_final": self.expected_final,
        }

    def _write_event_log(self):
        path = os.path.join(self.save_dir, "data", f"episode{self.ep_num}_event_log.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            **self._lh_metadata(),
            "episode": int(self.ep_num),
            "frames_total": int(self.FRAME_IDX),
            "stage_records": self.stage_records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------- button plumbing
    def get_current_button_value(self, button_actor, joint_name="button_joint"):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        idx = [j.get_name() for j in joints].index(joint_name)
        return art.get_qpos()[idx]

    def set_button_unpressed(self, button_actor, joint_name="button_joint", target=0.0):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        idx = [j.get_name() for j in joints].index(joint_name)
        qpos = art.get_qpos()
        qpos[idx] = target
        art.set_qpos(qpos)
        joints[idx].set_drive_target(target)

    def check_button_pressed(self, button_actor, joint_name="button_joint", threshold=-0.005):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        idx = [j.get_name() for j in joints].index(joint_name)
        return art.get_qpos()[idx] < threshold

    def update_button_reset(self, button_actor, flag_attr, joint_name="button_joint", threshold=-0.001):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        idx = [j.get_name() for j in joints].index(joint_name)
        if art.get_qpos()[idx] > threshold:
            setattr(self, flag_attr, False)

    def update_press_success(self, button_actor, flag_attr, cnt_attr):
        if self.check_button_pressed(button_actor) and not getattr(self, flag_attr):
            setattr(self, flag_attr, True)
            setattr(self, cnt_attr, getattr(self, cnt_attr) + 1)

    # -------------------------------------------------------------- success
    def check_success(self):
        self.update_button_reset(self.button, "press_flag")
        self.update_press_success(self.button, "press_flag", "press_cnt")
        self.set_button_unpressed(
            self.button, target=min(0.0, self.get_current_button_value(self.button) + 0.002))
        self._update_stage_tracker()
        self.max_reward = max(self.max_reward, self._tracker_ptr / self.n_stages)
        def fail(reason):
            # verbose only on the single post-play call (search/replay);
            # stage_records stays empty during eval rollouts
            if self.stage_records:
                print(f"[long_horizon] seed {self._lh_seed} FINAL CHECK FAILED: {reason}", flush=True)
            return False

        if self._script_fail:
            return False
        if self._tracker_ptr < self.n_stages:
            return fail(f"tracker at {self._tracker_ptr}/{self.n_stages}")
        for color, mat_name in self.expected_final["block_loc"].items():
            if not self._on_mat(self.blocks[color], mat_name):
                p = self.blocks[color].get_pose().p
                return fail(f"{color} block not on {mat_name} mat (at {p[:2].round(3).tolist()})")
        for ci in range(len(self.covers)):
            if not self._cover_parked(ci):
                p = self.covers[ci].get_pose().p
                return fail(f"cover{ci} not parked (at {p.round(3).tolist()})")
        if self.press_cnt != self.expected_press_total:
            return fail(f"press_cnt {self.press_cnt} != {self.expected_press_total}")
        if not self.is_right_gripper_open():
            return fail("right gripper not open")
        self.max_reward = max(self.max_reward, 1.0)
        return True
