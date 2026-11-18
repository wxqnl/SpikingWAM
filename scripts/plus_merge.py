"""Merge LIBERO-Plus runs that were split across GPUs and compare them with the state-input run on the same tasks.

  python scripts/plus_merge.py --tags percept_dr160o_full10,percept_dr160o_fullg,percept_dr160o_fullso --name dr160o

Prints score and episode node error (median, 90th percentile, cm) per category and suite, and the same for the tasks
inside and outside the 1050-task sample given by --sample. Writes runs/perception/checks/plus_<name>_full.json.
"""

import argparse
import collections
import json
import os

import numpy as np

BASE = "runs/f2tm16_geo_finecur16_textmean/libero_plus_ckpt_60000_ema_k1_steps1_fixed"


def load(path):
    return {(r["suite"], r["task_index"]): r for r in map(json.loads, open(path))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", required=True, help="comma list of run tags")
    p.add_argument("--name", required=True)
    p.add_argument("--base", default=BASE, help="the state-input run; tagged runs are <base>_<tag>")
    p.add_argument("--sample", default="percept_dr160o_sample", help="tag of a run on the 1050-task sample")
    a = p.parse_args()
    est = {}
    for t in a.tags.split(","):
        est.update(load(f"{a.base}_{t}/episodes.jsonl"))
    state = load(f"{a.base}/episodes.jsonl")
    keys = sorted(est)
    errors = [k for k in keys if "error" in est[k]]
    print(f"{len(keys)} of {len(state)} tasks, {len(errors)} with errors", collections.Counter(est[k]["category"] for k in errors))

    def row(ks):
        cm = [est[k]["percept_cm"] for k in ks if "percept_cm" in est[k]]
        return {"n": len(ks), "state": float(100 * np.mean([state[k]["success"] for k in ks])),
                "est": float(100 * np.mean([est[k]["success"] for k in ks])),
                "p50": float(np.median(cm)), "p90": float(np.percentile(cm, 90))}

    out = {"all": row(keys)}
    for c in sorted({est[k]["category"] for k in keys}):
        out[c] = row([k for k in keys if est[k]["category"] == c])
    for s in sorted({k[0] for k in keys}):
        out[s] = row([k for k in keys if k[0] == s])
    sample = load(f"{a.base}_{a.sample}/episodes.jsonl")
    out["inside sample"] = row([k for k in keys if k in sample])
    out["outside sample"] = row([k for k in keys if k not in sample])
    for n, v in out.items():
        print(f"{n:24s} n={v['n']:5d} state {v['state']:5.1f}  est {v['est']:5.1f}  err {v['p50']:.2f} / {v['p90']:.2f}")
    path = f"runs/perception/checks/plus_{a.name}_full.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
