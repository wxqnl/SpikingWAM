"""RoboTwin 2.0 side helpers (run with the RoboTwin python, cwd = the RoboTwin root).

`joints` reads the joint vector the actions are made of. `make_args` repeats the argument setup of RoboTwin's
script/collect_data.py and script/eval_policy.py for one task and task config (demo_clean = Easy, demo_randomized =
Hard); `task_actors` lists the task's objects: every attribute of the task that is an Actor / ArticulationActor (or a
list of them), in attribute-name order, without the scenery (wall, table) and the clutter of the randomized setting.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import yaml

from envs import CONFIGS_PATH
from envs.utils.actor_utils import Actor, ArticulationActor


SCENERY = {"wall", "table"}


def joints(task):
    """Measured joint vector of both arms (6 joints + gripper each): RoboTwin's `joint_action/vector`."""
    return np.asarray(task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState(), np.float32)


def make_task(task_name):
    return getattr(importlib.import_module(f"envs.{task_name}"), task_name)()


def make_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"], args["task_config"] = task_name, task_config
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        types = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        cams = yaml.load(f.read(), Loader=yaml.FullLoader)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = cams[head]["h"], cams[head]["w"]
    (emb,) = args["embodiment"]
    robot_file = types[emb]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        emb_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["left_embodiment_config"] = args["right_embodiment_config"] = emb_cfg
    args["embodiment_name"] = emb
    return args


def task_actors(task):
    """[(name, Actor)] of the task objects, sorted by attribute name; list attributes give name_0, name_1, ..."""
    out = []
    for name in sorted(vars(task)):
        if "clutter" in name or name in SCENERY:
            continue
        val = getattr(task, name)
        if isinstance(val, (Actor, ArticulationActor)):
            out.append((name, val))
        elif isinstance(val, (list, tuple)) and val and all(isinstance(v, (Actor, ArticulationActor)) for v in val):
            out += [(f"{name}_{i}", v) for i, v in enumerate(val)]
    return out
