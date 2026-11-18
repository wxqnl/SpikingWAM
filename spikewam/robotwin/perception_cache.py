"""Images for the RoboTwin front end (RoboTwin python, cwd = the RoboTwin root).

Runs RoboTwin's expert like spikewam.robotwin.collect and stores, at every --every-th data frame of an expert success, the three
cameras (head, left wrist, right wrist) resized to SIZE x SIZE, the world-to-pixel matrix of each on a 256 x 256 grid, and the
skeleton with its node descriptors. Training seeds only (far below the evaluation seeds, 100000 and up).

  cd $RT_ROOT && $RT_PY $SW_ROOT/spikewam/robotwin/perception_cache.py --task place_a2b_left --config demo_randomized --out $OUT
Writes <out>/<config>_<task>/{rgb.npy, meta.npz}.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import types

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "description", "utils"))
sys.path.insert(2, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from spikewam.robotwin.rt_env import make_args, make_task  # noqa: E402
from spikewam.robotwin.skeleton import RTSkeleton  # noqa: E402

CAMS = ("head_camera", "left_camera", "right_camera")
SIZE = 192
GRID = 256                                    # the pixel coordinates of the matrices (spikewam.perception.IMAGE)
N_TEXT = 8                                    # instruction variants kept per episode


def views(task):
    """Images (3, SIZE, SIZE, 3) uint8 and world-to-pixel matrices (3, 3, 4)."""
    task._update_render()
    task.cameras.update_picture()
    rgb, cfg = task.cameras.get_rgb(), task.cameras.get_config()
    images, mats = [], []
    for c in CAMS:
        im = rgb[c]["rgb"]
        h, w = im.shape[:2]
        images.append(cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        m = cfg[c]["intrinsic_cv"] @ np.asarray(cfg[c]["extrinsic_cv"])[:3]
        mats.append(np.diag([GRID / w, GRID / h, 1.0]) @ m)
    return np.stack(images), np.array(mats, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", default="demo_randomized")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--every", type=int, default=3, help="keep every this many data frames")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=12600, help="seconds; stop collecting and save what there is")
    p.add_argument("--retry", action="store_true", help="take the task even if its directory exists (an earlier run was killed)")
    a = p.parse_args()
    d = os.path.join(a.out, f"{a.config}_{a.task}")
    if os.path.exists(os.path.join(d, "meta.npz")):
        print("done already", d, flush=True)
        return
    try:
        os.makedirs(d, exist_ok=a.retry)      # atomic claim, so two queues never run the same task
    except FileExistsError:
        print("taken by another process", d, flush=True)
        return
    task = make_task(a.task)
    args = make_args(a.task, a.config)
    args["need_plan"], args["save_data"], args["render_freq"] = True, False, 0
    state = {"skel": None, "n": 0, "frames": []}

    def take_picture(self):                   # replaces Base_Task._take_picture
        if state["n"] % a.every == 0:
            images, mats = views(self)
            state["frames"].append((state["n"], images, mats, state["skel"]()))
        state["n"] += 1

    task._take_picture = types.MethodType(take_picture, task)
    eps, seed, fails, no_planner, t0 = [], a.seed0, 0, 0, time.time()
    while len(eps) < a.episodes and time.time() - t0 < a.budget:
        state["n"], state["frames"] = 0, []
        ok = False
        try:
            task.setup_demo(now_ep_num=len(eps), seed=seed, **args)
            state["skel"] = RTSkeleton(task)
            info = task.play_once()
            ok = bool(task.plan_success and task.check_success())
        except Exception as e:                # RoboTwin's collector counts these as failed seeds too
            print(f"seed {seed}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            no_planner = no_planner + 1 if "planner" in str(e) else 0
            if no_planner >= 20:
                raise SystemExit(f"{no_planner} seeds in a row without a motion planner")
        if ok and state["frames"]:
            sk = state["skel"]
            desc = generate_episode_descriptions(a.task, [info["info"]], 100)[0]
            seen = list(desc["seen"])
            text = [seen[i % len(seen)] for i in range(N_TEXT)]
            kinds = ["robot"] * sk.n_nodes
            for i, s in enumerate(sk.slot):
                if s:
                    kinds[i] = re.sub(r"_\d+$", "", sk.object_names[s - 1])
            eps.append({"seed": seed, "frames": state["frames"], "node_id": sk.node_id, "role": sk.role, "slot": sk.slot, "kinds": kinds, "text": text})
            print(f"episode {len(eps)}: seed {seed}, {len(state['frames'])} frames, {sk.n_nodes} nodes ({time.time() - t0:.0f} s)", flush=True)
        else:
            fails += 1
        task.close_env(clear_cache=((seed + 1) % args["clear_cache_freq"] == 0))
        seed += 1
    n_frames = sum(len(e["frames"]) for e in eps)
    n_pad = max(len(e["node_id"]) for e in eps)
    rgb = np.lib.format.open_memmap(os.path.join(d, "rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(n_frames, 3, SIZE, SIZE, 3))
    skel = np.zeros((n_frames, n_pad, 3), np.float32)
    desc = {k: np.zeros((n_frames, n_pad), np.int64) for k in ("node_id", "role", "slot")}
    M, ep_i, t_i, n_nodes = np.zeros((n_frames, 3, 3, 4), np.float32), np.zeros(n_frames, np.int64), np.zeros(n_frames, np.int64), np.zeros(n_frames, np.int64)
    vocab = sorted({k for e in eps for k in e["kinds"]})
    index = {k: i for i, k in enumerate(vocab)}
    kind = np.zeros((n_frames, n_pad), np.int64)
    i = 0
    for k, e in enumerate(eps):
        n = len(e["node_id"])
        for t, images, mats, s in e["frames"]:
            rgb[i], M[i], skel[i, :n], ep_i[i], t_i[i], n_nodes[i] = images, mats, s, k, t, n
            for name in desc:
                desc[name][i, :n] = e[name]
            kind[i, :n] = [index[x] for x in e["kinds"]]
            i += 1
    rgb.flush()
    del rgb
    os.replace(os.path.join(d, "rgb.tmp.npy"), os.path.join(d, "rgb.npy"))
    np.savez(os.path.join(d, "meta.npz"), skel=skel, M=M, ep=ep_i, t=t_i, n_nodes=n_nodes, kind=kind, vocab=np.array(vocab), seeds=np.array([e["seed"] for e in eps]),
             language=np.array([e["text"] for e in eps]), **desc)
    print(f"{a.config} {a.task}: {len(eps)} episodes, {fails} failed seeds, {n_frames} frames, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
