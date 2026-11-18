"""LIBERO task lookup and simulator construction without cameras (the sim-state skeleton needs no rendering)."""

from __future__ import annotations

import os

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# Episode step limits of the standard LIBERO evaluation (OpenVLA protocol).
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10


def get_suite(name):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[name]()


def bddl_path(task):
    from libero.libero import get_libero_path
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def make_env(bddl_file, **kwargs):
    """A LIBERO ControlEnv with no camera observations and no renderer."""
    from libero.libero.envs.env_wrapper import ControlEnv
    kwargs.setdefault("use_camera_obs", False)
    kwargs.setdefault("has_offscreen_renderer", False)
    kwargs.setdefault("has_renderer", False)
    return ControlEnv(bddl_file_name=bddl_file, **kwargs)


def scene_objects(env):
    """(object and fixture names, obj_of_interest) of a LIBERO env."""
    inner = env.env
    return list(inner.objects_dict) + list(inner.fixtures_dict), list(inner.obj_of_interest)


def task_key(suite_name, task_id):
    return f"{suite_name}/{task_id}"
