"""D0 failure phases (research plan section 5.3): where in the task did each failed episode stop?

From the recorded skeletons of failed episodes (eval_libero --record), relative to the first object group (slot 1):
  approach   the grip site never came within NEAR of any slot-1 node;
  grasp      it came near, but no slot-1 node moved more than MOVED from where it started;
  transport  slot 1 moved (picked up, pushed, drawer pulled), but the task was not completed.
Also reports the closest grip-to-slot-1 distance of every failed episode.

  python scripts/failure_phases.py runs/<run>/traj_<name>.pkl [more.pkl ...]
"""

import pickle
import sys
from collections import Counter

import numpy as np

from spikewam.geometry import GRIP_NODE

NEAR, MOVED = 0.04, 0.02


def phase(rec):
    traj = rec["traj"].astype(np.float32)
    sel = rec["slot"] == 1
    grip = traj[:, GRIP_NODE]
    obj = traj[:, sel]
    closest = np.linalg.norm(obj - grip[:, None], axis=-1).min()
    moved = np.linalg.norm(obj - obj[0], axis=-1).max()
    if closest > NEAR:
        return "approach", closest
    if moved < MOVED:
        return "grasp", closest
    return "transport", closest


def main():
    for path in sys.argv[1:]:
        recs = pickle.load(open(path, "rb"))
        by_suite = {}
        for r in recs:
            p, d = phase(r)
            by_suite.setdefault(r["suite"], []).append((p, d))
        print(path)
        for s, rows in sorted(by_suite.items()):
            c = Counter(p for p, _ in rows)
            ds = np.array([d for _, d in rows]) * 1000
            print(f"  {s:15s} failed {len(rows):4d}: " + ", ".join(f"{k} {c[k]}" for k in ("approach", "grasp",
                                                                                           "transport"))
                  + f" | closest grip-object distance median {np.median(ds):.0f} mm")


if __name__ == "__main__":
    main()
