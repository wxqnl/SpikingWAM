"""Paired RoboTwin success table from spikewam/robotwin/evaluate.py results (implementation record 4.11).

Reads every runs/robotwin_eval/<dir>/<config>/<task>.json (separate per-run dirs and paired <run1>+<run2> dirs alike),
and per task compares the runs on the seeds all of them have finished (same seed = same scene and instruction).

  python scripts/robotwin_table.py --config demo_clean --runs rt_r0 rt_f2 [--full] [--instr seen]
--full: only tasks every run has finished (--test_num seeds); otherwise partial tasks count on their common seeds.
--instr seen: the result dirs with the suffix _seen (setup/robotwin_eval.sh INSTR=seen); by default the others.
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", default="runs/robotwin_eval")
    pa.add_argument("--config", default="demo_clean")
    pa.add_argument("--runs", nargs="+", required=True)
    pa.add_argument("--test_num", type=int, default=100)
    pa.add_argument("--full", action="store_true")
    pa.add_argument("--instr", default="unseen", choices=("unseen", "seen"))
    a = pa.parse_args()
    per = {}                                     # task -> run -> {seed: success}
    for path in glob.glob(os.path.join(a.root, "*", a.config, "*.json")):
        if path.split(os.sep)[-3].endswith("_seen") != (a.instr == "seen"):
            continue
        d = json.load(open(path))
        for run, recs in d["results"].items():
            if run in a.runs:
                per.setdefault(d["task"], {}).setdefault(run, {}).update(
                    {s: r["success"] for s, r in zip(d["seeds"], recs)})
    rows = []
    for task in sorted(per):
        if any(r not in per[task] for r in a.runs):
            continue
        seeds = set.intersection(*(set(per[task][r]) for r in a.runs))
        if not seeds or (a.full and len(seeds) < a.test_num):
            continue
        rows.append((task, len(seeds), [np.mean([per[task][r][s] for s in seeds]) * 100 for r in a.runs]))
    print(f"{'task':32s} {'n':>4s} " + " ".join(f"{r:>8s}" for r in a.runs))
    for task, n, sr in rows:
        print(f"{task:32s} {n:4d} " + " ".join(f"{v:8.1f}" for v in sr))
    m = np.array([sr for _, _, sr in rows])
    print(f"{len(rows)} tasks ({sum(n < a.test_num for _, n, _ in rows)} partial), mean "
          + " ".join(f"{r} {v:.1f}" for r, v in zip(a.runs, m.mean(0))))
    if len(a.runs) == 2:
        d = m[:, 1] - m[:, 0]
        print(f"{a.runs[1]} - {a.runs[0]}: {d.mean():+.1f}, better on {(d > 0).sum()} tasks, worse on {(d < 0).sum()}, "
              f"tied {(d == 0).sum()}")


if __name__ == "__main__":
    main()
