"""How do RoboTwin rollouts fail? (implementation record 4.13)

Reads the trajectories of a recorded paired evaluation (setup/robotwin_eval.sh with RECORD=1: per run and seed, the
joint vector, skeleton and executed joint targets of every plan) and reports per task and run, over the failed
episodes:
  stall      plans at the end of the episode in a row whose arm joints moved < STALL degrees (stuck)
  grasps     gripper closings (measured gripper crossing the middle of its range downwards)
  lift       episodes in which some object centre rose more than LIFT cm
  reach      closest approach of a gripper (grip-site node) to an object centre, cm (median)
  track      largest arm-joint gap between the last commanded target of a plan and the joint vector measured at the
             next plan, degrees (median over plans; a blocked arm leaves a large gap)
  jump       largest arm-joint gap between the first commanded target and the measured joint vector (median, deg)
and the same for the seeds that the reference run (first run) solved and the others did not.

  python scripts/rt_failure.py --dir runs/robotwin_eval/rt_r0_120k+rt_f2_120k+rt_f2ppt_120k_rec/demo_clean
"""

import argparse
import glob
import os
import pickle

import numpy as np

ARM = [i for i in range(14) if i not in (6, 13)]
GRIP_Q = (6, 13)
GRIP_NODE = (7, 17)                                    # grip-site node of the left / right arm (layout: 10 per arm)
CENTER = 3                                             # role of an object centre node
STALL, LIFT = 1.0, 3.0


def stats(ep):
    q, skel, act = ep["q"].astype(np.float64), ep["skel"].astype(np.float64), ep["act"].astype(np.float64)
    move = np.degrees(np.abs(np.diff(q[:, ARM], axis=0)).max(1)) if len(q) > 1 else np.zeros(0)
    stall = 0
    for m in move[::-1]:
        if m >= STALL:
            break
        stall += 1
    g = q[:, GRIP_Q]
    mid = (g.max() + g.min()) / 2 if g.max() - g.min() > 0.1 else -1
    grasps = int(((g[:-1] > mid) & (g[1:] <= mid)).sum()) if len(g) > 1 else 0
    centres = np.nonzero(ep["role"] == CENTER)[0]
    z = skel[:, centres, 2]
    lift = bool(len(centres) and (z - z[:1]).max() * 100 > LIFT)
    hands = skel[:, list(GRIP_NODE)]
    reach = (np.linalg.norm(hands[:, :, None] - skel[:, None, centres], axis=-1).min() * 100) if len(centres) else np.nan
    track = np.degrees(np.abs(q[1:, ARM] - act[:-1, -1][:, ARM]).max(1)) if len(q) > 1 else np.zeros(1)
    jump = np.degrees(np.abs(act[:, 0][:, ARM] - q[:, ARM]).max(1))
    return dict(stall=stall, plans=len(q), grasps=grasps, lift=lift, reach=reach, track=np.median(track),
                jump=np.median(jump))


def line(name, eps):
    if not eps:
        return f"  {name:18s} -"
    s = [stats(e) for e in eps]
    f = lambda k: np.array([x[k] for x in s], np.float64)
    return (f"  {name:18s} n {len(s):3d}  stall {f('stall').mean():5.1f}/{f('plans').mean():4.1f} plans "
            f"(stuck >= 5: {(f('stall') >= 5).mean() * 100:3.0f}%)  grasps {f('grasps').mean():4.1f} "
            f"(none {(f('grasps') == 0).mean() * 100:3.0f}%)  lift {f('lift').mean() * 100:3.0f}%  "
            f"reach {np.nanmedian(f('reach')):5.1f} cm  track {np.median(f('track')):5.2f}  jump {np.median(f('jump')):5.2f}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dir", required=True)
    a = pa.parse_args()
    for path in sorted(glob.glob(os.path.join(a.dir, "traj_*.pkl"))):
        with open(path, "rb") as f:
            traj = pickle.load(f)
        runs = list(traj)
        seeds = sorted(set.intersection(*[set(traj[r]) for r in runs]))
        task = os.path.basename(path)[5:-4]
        print(f"{task} ({len(seeds)} seeds): " + "  ".join(
            f"{r} {np.mean([traj[r][s]['success'] for s in seeds]) * 100:.0f}%" for r in runs))
        for r in runs:
            print(line(f"{r} all", [traj[r][s] for s in seeds]))
            print(line(f"{r} failed", [traj[r][s] for s in seeds if not traj[r][s]["success"]]))
        ref = runs[0]
        for r in runs[1:]:
            lost = [s for s in seeds if traj[ref][s]["success"] and not traj[r][s]["success"]]
            print(line(f"{r} lost vs {ref[:6]}", [traj[r][s] for s in lost]))
            print(line(f"{ref[:6]} on them", [traj[ref][s] for s in lost]))


if __name__ == "__main__":
    main()
