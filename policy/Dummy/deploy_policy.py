import numpy as np


class DummyModel:
    """Trivial policy: holds the current joint configuration."""
    pass


def encode_obs(observation):
    return observation


def get_model(usr_args):
    return DummyModel()


def eval(TASK_ENV, model, observation):
    # Hold pose: command the current 14-dim joint state as the action.
    action = np.array(observation["joint_action"]["vector"])
    TASK_ENV.take_action(action, action_type="qpos")


def reset_model(model):
    pass
