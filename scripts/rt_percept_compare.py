"""Compare RoboTwin closed-loop runs on the same seeds: totals, paired counts, and the tasks that differ most.

  python scripts/rt_percept_compare.py rt160g_rigid rt160g [--config demo_clean] [--state]
A tag names runs/robotwin_eval/rt_f2ppt16_big_percept_<tag>; --state adds the state-input run (rt_f2ppt16_big).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "robotwin_eval")


def load(tag, config):
    d = os.path.join(ROOT, tag if tag == "rt_f2ppt16_big" else f"rt_f2ppt16_big_percept_{tag}", config)
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        eps = next(iter(r["results"].values()))
        out[r["task"]] = {s: e["success"] for s, e in zip(r["seeds"], eps)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--state", action="store_true")
    p.add_argument("--top", type=int, default=12)
    a = p.parse_args()
    tags = a.tags + (["rt_f2ppt16_big"] if a.state else [])
    runs = {t: load(t, a.config) for t in tags}
    for t, r in runs.items():
        n = sum(len(v) for v in r.values())
        print(f"{t:>20}: {sum(sum(v.values()) for v in r.values())}/{n} ({len(r)} tasks)")
    base = tags[0]
    for other in tags[1:]:
        common = [t for t in runs[base] if t in runs[other]]
        win = {t: [0, 0, 0] for t in common}
        for t in common:
            for s, ok in runs[base][t].items():
                if s in runs[other][t]:
                    win[t][0] += ok
                    win[t][1] += runs[other][t][s]
                    win[t][2] += 1
        A, B, N = (sum(v[i] for v in win.values()) for i in range(3))
        print(f"paired {base} vs {other}: {A} vs {B} of {N} episodes ({len(common)} tasks)")
        diff = sorted(common, key=lambda t: win[t][0] - win[t][1])
        print("  worse:", ", ".join(f"{t} {win[t][0]}-{win[t][1]}" for t in diff[:a.top] if win[t][0] < win[t][1]))
        print("  better:", ", ".join(f"{t} {win[t][0]}-{win[t][1]}" for t in diff[::-1][:a.top] if win[t][0] > win[t][1]))


if __name__ == "__main__":
    main()
