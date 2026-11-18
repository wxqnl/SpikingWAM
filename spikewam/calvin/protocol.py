"""The official CALVIN long-horizon evaluation (calvin_models/calvin_agent/evaluation/evaluate_policy.py): 1,000
chains of 5 subtasks from get_sequences(1000), each started from get_env_state_for_initial_condition, 360 steps per
subtask, the first validation annotation of each subtask as the instruction. Both functions are taken from the CALVIN
checkout ($CALVIN_ROOT, default /mnt/data/user01/calvin) without its training stack: get_sequences is imported with a
stand-in for calvin_agent.evaluation.utils (it only needs temp_seed), and the initial state is rebuilt here with an
FNV-1 32-bit hash in place of pyhash.fnv1_32 (pyhash no longer builds; test vectors in fnv1_32).

  python -m spikewam.calvin.protocol [--out data/calvin/eval_sequences.json]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import types

import numpy as np

CALVIN_ROOT = os.environ.get("CALVIN_ROOT", "/mnt/data/user01/calvin")
EP_LEN = 360
NUM_SEQUENCES = 1000


def fnv1_32(s):
    """FNV-1 (32 bit) of the UTF-8 bytes of s, as pyhash.fnv1_32()(s): "" -> 0x811c9dc5, "a" -> 0x050c5d7e,
    "foobar" -> 0x31f0b262."""
    h = 0x811C9DC5
    for b in s.encode():
        h = (h * 0x01000193) & 0xFFFFFFFF
        h ^= b
    return h


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_sequences(num_sequences=NUM_SEQUENCES):
    """calvin_agent.evaluation.multistep_sequences.get_sequences, unchanged."""
    sys.path.insert(0, os.path.join(CALVIN_ROOT, "calvin_models"))
    utils = types.ModuleType("calvin_agent.evaluation.utils")
    utils.temp_seed = temp_seed
    sys.modules["calvin_agent.evaluation.utils"] = utils
    from calvin_agent.evaluation.multistep_sequences import get_sequences as official
    return official(num_sequences)


def initial_state(initial_condition):
    """calvin_agent.evaluation.utils.get_env_state_for_initial_condition with fnv1_32 for the hasher."""
    from numpy import pi
    robot_obs = np.array([0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596, 1.50013585, 0.07999963,
                          -1.21779124, 1.03987629, 2.11978254, -2.34205014, -0.87015899, 1.64119093, 0.55344928, 1.0])
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
                   np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01])]
    seed = fnv1_32(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_DATA", "data"), "calvin", "eval_sequences.json"))
    a = p.parse_args()
    out = []
    for cond, seq in get_sequences():
        robot_obs, scene_obs = initial_state(cond)
        out.append({"condition": dict(cond), "sequence": list(seq), "robot_obs": robot_obs.tolist(),
                    "scene_obs": scene_obs.tolist()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"{len(out)} sequences -> {a.out}; first: {out[0]['sequence']}")


if __name__ == "__main__":
    main()
