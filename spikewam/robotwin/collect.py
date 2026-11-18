"""Expert demonstrations with sim-state skeletons for RoboTwin 2.0 (RoboTwin python, cwd = the RoboTwin root).

Like RoboTwin's script/collect_data.py (seeds 0, 1, 2, ... until --episodes expert successes), in one pass: the
expert episode itself is recorded at RoboTwin's data frames (every save_freq simulation steps, where RoboTwin's
collector saves an observation) as the skeleton (spikewam.robotwin.skeleton) and the joint vector of both arms
(6 joints + gripper each, RoboTwin's `joint_action/vector`), plus one frame at the end. The action of frame t is the
joint vector of frame t + 1 (as in RoboTwin's DP / ACT data processing); the last frame holds its own.
Instructions: RoboTwin's episode descriptions for the objects the episode drew (all seen and unseen ones are kept;
training uses the first seen one).

  cd $RT_ROOT && $RT_PY $SW_ROOT/spikewam/robotwin/collect.py --task place_a2b_left --out $OUT [--episodes 50]
Writes <out>/<task>.npz. A slow task can be split into parts with --seed0 / --tag (seeds from seed0, written to
<out>/<task><tag>.npz) and merged with scripts/robotwin_merge.py; all training seeds stay far below the evaluation
seeds (100000 and up).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "description", "utils"))
sys.path.insert(2, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from spikewam.robotwin.rt_env import joints, make_args, make_task  # noqa: E402
from spikewam.robotwin.skeleton import RTSkeleton  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    task = make_task(a.task)
    args = make_args(a.task, a.config)
    args["need_plan"], args["save_data"], args["render_freq"] = True, False, 0
    frames = []
    state = {"skel": None}

    def take_picture(self):                     # replaces Base_Task._take_picture: record instead of saving
        frames.append((state["skel"](), joints(self)))

    task._take_picture = types.MethodType(take_picture, task)
    eps, seed, fails, t0 = [], a.seed0, 0, time.time()
    no_planner = 0          # a planner that failed to build (e.g. no GPU memory at start) fails every later seed
    while len(eps) < a.episodes:
        frames.clear()
        ok = False
        try:
            task.setup_demo(now_ep_num=len(eps), seed=seed, **args)
            state["skel"] = RTSkeleton(task)
            info = task.play_once()
            ok = bool(task.plan_success and task.check_success())
        except Exception as e:                  # RoboTwin's collector counts these as failed seeds too
            print(f"seed {seed}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            no_planner = no_planner + 1 if "planner" in str(e) else 0
            if no_planner >= 20:                # exit, so that the crash watcher frees the lock and runs the part again
                raise SystemExit(f"{no_planner} seeds in a row without a motion planner")
        if ok and len(frames) > 1:
            frames.append((state["skel"](), joints(task)))
            sk = state["skel"]
            desc = generate_episode_descriptions(a.task, [info["info"]], 100)[0]
            eps.append({"seed": seed, "skel": np.stack([f[0] for f in frames]), "q": np.stack([f[1] for f in frames]),
                        "node_id": sk.node_id, "role": sk.role, "slot": sk.slot, "objects": sk.object_names,
                        "info": info, "seen": list(desc["seen"]), "unseen": list(desc["unseen"])})
            print(f"episode {len(eps)}: seed {seed}, {len(frames)} frames, {sk.n_nodes} nodes "
                  f"({time.time() - t0:.0f} s)", flush=True)
        else:
            fails += 1
        task.close_env(clear_cache=((seed + 1) % args["clear_cache_freq"] == 0))
        seed += 1
    n_pad = max(e["skel"].shape[1] for e in eps)
    lens = np.array([len(e["q"]) for e in eps])
    skel = np.zeros((lens.sum(), n_pad, 3), np.float32)
    desc = np.zeros((3, len(eps), n_pad), np.int64)
    acts = []
    st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    for i, e in enumerate(eps):
        n = e["skel"].shape[1]
        skel[st[i]:st[i] + lens[i], :n] = e["skel"]
        for k, key in enumerate(("node_id", "role", "slot")):
            desc[k, i, :n] = e[key]
        acts.append(np.concatenate([e["q"][1:], e["q"][-1:]]))
    np.savez_compressed(
        os.path.join(a.out, f"{a.task}{a.tag}.npz"), skel=skel, actions=np.concatenate(acts).astype(np.float32),
        n_nodes=np.array([e["skel"].shape[1] for e in eps]), node_id=desc[0], role=desc[1], slot=desc[2],
        ep_start=st, ep_len=lens, seeds=np.array([e["seed"] for e in eps]),
        language=np.array([e["seen"][0] for e in eps], dtype=object),
        meta=np.array([json.dumps({"objects": e["objects"], "info": e["info"], "seen": e["seen"],
                                   "unseen": e["unseen"]}, default=str) for e in eps], dtype=object))
    print(f"{a.task}: {len(eps)} episodes, {fails} failed seeds, {lens.sum()} frames, {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
