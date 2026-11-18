"""The geometry query must be the same in closed loop and in training: in the simulator, compare RTFrontEnd.model_points (read from
RTSkeleton.local) with train_perception.model_frame_points (recovered from the true skeleton) on the same scenes (RoboTwin python,
cwd = the RoboTwin root).

  cd $RT_ROOT && CUDA_VISIBLE_DEVICES=6 $RT_PY $SW_ROOT/scripts/rt_geo_consistency.py place_shoe blocks_ranking_size open_laptop
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(2, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from spikewam.robotwin.layout import N_ROBOT  # noqa: E402
from spikewam.robotwin.percept import RTFrontEnd  # noqa: E402
from spikewam.robotwin.rt_env import make_args, make_task  # noqa: E402
from spikewam.robotwin.skeleton import RTSkeleton  # noqa: E402
from train_perception import model_frame_points  # noqa: E402


def main():
    for name in sys.argv[1:]:
        task = make_task(name)
        args = make_args(name, "demo_clean")
        args.update(render_freq=0)
        for seed in (5000, 5001):
            try:
                task.setup_demo(now_ep_num=0, seed=seed, **args)
            except Exception as e:
                print(name, seed, type(e).__name__)
                task.close_env()
                continue
            sk = RTSkeleton(task)
            skel = np.asarray(sk(), np.float32)
            closed = RTFrontEnd.model_points(sk)
            obj = slice(N_ROBOT, None)
            t = lambda x, d: torch.as_tensor(np.asarray(x)[obj][None], dtype=d)
            train = model_frame_points(t(skel, torch.float32), t(sk.node_id, torch.long), t(sk.role, torch.long),
                                       torch.ones(1, len(skel) - N_ROBOT, dtype=torch.bool))[0].numpy()
            both = (closed[:, 3] > 0) & (train[:, 3] > 0)
            only = int(((closed[:, 3] > 0) != (train[:, 3] > 0)).sum())
            err = np.abs(closed[both, :3] - train[both, :3]).max() * 100 if both.any() else float("nan")
            print(f"{name} seed {seed}: {int(both.sum())} nodes with points in both, {only} in one only, max difference {err:.3f} cm", flush=True)
            task.close_env()


if __name__ == "__main__":
    main()
