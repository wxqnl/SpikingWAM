"""How much of the joint target k steps ahead follows the current posture, joint by joint? (implementation record 4.13)

No training. Per task, a linear regression of action[t + k] (each arm joint) on the current joint vector
q_t = action[t - 1] (12 arm joints) and the object node positions at t (slot > 0), within episodes; the slope on the
joint's own q is the part of the target that moves with the current posture, the rest is set by the scene. Prints the
median slope over the 50 tasks per joint and chunk step, and the target spread (std over all frames, degrees) for
three anchors: none (absolute), q_t (posture-relative) and w * q_t with the median slope w.

  python scripts/rt_anchor_slope.py [--steps 0 3 7 11 15 31]
"""

import argparse
import glob
import os

import numpy as np

ARM = [i for i in range(14) if i not in (6, 13)]


def task_rows(path, k):
    """(q_t, object positions, target action[t + k]) over the frames of one task, episodes of its usual node count."""
    z = np.load(path, allow_pickle=True)
    act, skel = z["actions"], z["skel"]
    n_usual = np.bincount(z["n_nodes"]).argmax()
    q, obj, y = [], [], []
    for e, (st, ln) in enumerate(zip(z["ep_start"], z["ep_len"])):
        if z["n_nodes"][e] != n_usual:
            continue
        t = np.arange(st + 1, st + ln - k)
        sel = z["slot"][e, :n_usual] > 0
        q.append(act[t - 1][:, ARM])
        obj.append(skel[t][:, :n_usual][:, sel].reshape(len(t), -1))
        y.append(act[t + k][:, ARM])
    return np.concatenate(q), np.concatenate(obj), np.concatenate(y)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--steps", type=int, nargs="+", default=[0, 3, 7, 11, 15, 31])
    a = pa.parse_args()
    paths = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    print("joints: left arm 1-6 | right arm 1-6")
    for k in a.steps:
        slopes, rows = [], []
        for p in paths:
            q, obj, y = task_rows(p, k)
            x = np.concatenate([q, obj, np.ones((len(q), 1))], 1)
            coef = np.linalg.solve(x.T @ x + 1e-4 * np.eye(x.shape[1]), x.T @ y)
            slopes.append(np.diag(coef[:12]))
            rows.append((q, y))
        w = np.median(slopes, 0)
        q, y = (np.concatenate(r) for r in zip(*rows))
        sd = lambda v: " ".join(f"{s:5.1f}" for s in np.degrees(v.std(0)))
        print(f"k={k:2d} median own slope " + " ".join(f"{v:5.2f}" for v in w))
        print(f"      target std (deg)  absolute {sd(y)}")
        print(f"                        a - q_t  {sd(y - q)}")
        print(f"                        a - w q  {sd(y - w * q)}")


if __name__ == "__main__":
    main()
