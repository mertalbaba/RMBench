"""ToM-WM policy adapter for RMBench eval (client side of tom_wm/runtime/server.py).

Runs inside the RMBench sim env: stdlib + numpy only. All model work happens
on the WM server (cosmos env, launched separately — see tom_wm/runtime/server.py
docstring); this adapter ships observations over the RMBench socket protocol
and steps the returned 16-step qpos chunk.

Clip framing (must match tom_wm/runtime/tick.py): the first eval() of an
episode sends the single current head frame; every later eval() sends 17
frames — the last frame of the previous chunk plus the 16 frames its
execution produced. The current third-person frame and qpos ride along.
"""

import os
import sys

import numpy as np

TOM_WM_ROOT = os.environ.get("TOM_WM_ROOT", "/home/mealbaba/tom-wm")
if TOM_WM_ROOT not in sys.path:
    sys.path.insert(0, TOM_WM_ROOT)

from tom_wm.runtime.client import WMClient  # noqa: E402


class TomWMPolicy:
    def __init__(self, usr_args):
        self.client = WMClient(
            host=usr_args.get("host") or "localhost",
            port=int(usr_args.get("port") or 6001),
        )
        self.task_name = usr_args.get("task_name")
        self.frame_buffer = []  # head frames since the last tick

    @property
    def task_group(self):
        return f"rmbench/{self.task_name}" if self.task_name else None

    def reset(self):
        self.frame_buffer = []
        self.client.reset()


def encode_obs(observation):
    return {
        "head_rgb": np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8),
        "third_rgb": (np.asarray(observation["third_view_rgb"], dtype=np.uint8)
                      if "third_view_rgb" in observation else None),
        "qpos": np.asarray(observation["joint_action"]["vector"], dtype=np.float64),
    }


def get_model(usr_args):
    return TomWMPolicy(usr_args)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    if not model.frame_buffer:  # first tick of the episode: single-frame clip
        model.frame_buffer = [obs["head_rgb"]]
    clip = np.stack(model.frame_buffer)

    res = model.client.tick(
        frames=clip,
        qpos=obs["qpos"],
        task=instruction or "",
        task_group=model.task_group,
        third_frame=obs["third_rgb"],
    )
    qpos_chunk = np.asarray(res["qpos_chunk"], dtype=np.float64)  # (16, 14)

    # next clip starts from the frame this chunk begins on
    model.frame_buffer = [clip[-1]]
    for action in qpos_chunk:
        TASK_ENV.take_action(action, action_type="qpos")
        observation = TASK_ENV.get_obs()
        model.frame_buffer.append(encode_obs(observation)["head_rgb"])


def reset_model(model):
    model.reset()
