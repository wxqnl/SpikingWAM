"""How far did failed episodes get? Per-goal completion from the "goals" field of eval_libero results.

For tasks whose BDDL goal has several predicates (e.g. both objects in the basket, or the bowl in the drawer and the
drawer closed), a failed episode either completed none of them or some of them. Reports, per run, the success rate
of the multi-goal tasks and of the rest, and how the failed multi-goal episodes split into none / partial; plus the
completion rate of each goal predicate of each multi-goal task, and the completion rate of each predicate kind
(on / in / close / open / turnon) over all tasks.

  python scripts/goal_progress.py runs/r0_ann/eval_ckpt_60000_ema_goals.json runs/<run>/eval_..._goals.json
"""

import json
import sys
from collections import defaultdict

import numpy as np


def predicate_kinds():
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from spikewam.libero_tasks import SUITES, bddl_path, get_suite
    kinds = {}
    for s in SUITES:
        su = get_suite(s)
        for t in range(su.n_tasks):
            kinds[(s, t)] = [g[0] for g in robosuite_parse_problem(bddl_path(su.get_task(t)))["goal_state"]]
    return kinds


def main():
    kinds = predicate_kinds()
    for path in sys.argv[1:]:
        with open(path) as f:
            eps = json.load(f)["episodes"]
        multi = [e for e in eps if len(e["goals"]) > 1]
        single = [e for e in eps if len(e["goals"]) == 1]
        failed = [e for e in multi if not e["success"]]
        partial = sum(any(e["goals"]) for e in failed)
        print(path)
        print(f"  multi-goal tasks: {len(multi) // 50} tasks, success {np.mean([e['success'] for e in multi]):.3f}; "
              f"failed {len(failed)}: none {len(failed) - partial}, partial {partial}")
        print(f"  single-goal tasks: {len(single) // 50} tasks, success "
              f"{np.mean([e['success'] for e in single]):.3f}")
        per_task = defaultdict(list)
        for e in multi:
            per_task[f"{e['suite']}/{e['task_id']}"].append(e["goals"])
        for t, g in sorted(per_task.items()):
            rates = np.mean(np.array(g, dtype=float), 0)
            print(f"    {t:16s} goals " + " ".join(f"{r:.2f}" for r in rates))
        acc = defaultdict(list)
        for e in eps:
            for name, ok in zip(kinds[(e["suite"], e["task_id"])], e["goals"]):
                acc[name].append(ok)
        print("  predicates: " + "  ".join(f"{k} {np.mean(v) * 100:.1f} (n={len(v)})" for k, v in sorted(acc.items())))


if __name__ == "__main__":
    main()
