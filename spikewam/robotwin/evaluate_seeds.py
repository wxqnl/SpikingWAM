"""Closed-loop RoboTwin evaluation on given seeds and instructions, for splitting one task's episodes over processes
(RoboTwin python, cwd = the RoboTwin root).

The seeds file holds {"seeds": [...], "instructions": [...]}: seeds that the expert check of an earlier run accepted, with
that run's instruction for each. This script runs episodes a..b-1 of the file: the same scene reset, rollout, policy and
front-end settings as evaluate.py, without a second expert check (the seed is already known to be solvable) and with the
stored instruction. Writes <out>/parts/<config>/<task>.<a>-<b>.json (the layout of evaluate.py's result file, plus
"range" and "done"); a second call resumes it. evaluate_merge in scripts joins the parts into <out>/<config>/<task>.json.

  cd $RT_ROOT && $RT_PY $SW_ROOT/spikewam/robotwin/evaluate_seeds.py --task place_a2b_left --config demo_clean \
      --ckpt $SW_RUNS/rt_f2ppt16_big/ckpt_150000.pt --out <dir> --seeds_file <file> --range 10:40 --perception <front end>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from evaluate import make_args, make_task, rollout, SkeletonPolicy  # noqa: E402  (evaluate.py sets up the paths)
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
import random  # noqa: E402


EXPERT_FIRST = {"open_laptop", "place_object_scale", "put_object_cabinet"}    # check_success reads attributes set in play_once


def poisoned(e):
    """11-15: one CUDA out-of-memory at setup leaves the task without its planners ('Robot' object has no attribute
    'right_planner'), and every later setup fails the same way: the old loop then tried replacement seeds for 20 hours.
    Exit instead; eval_units.py retries the unit, which resumes from its saved episodes."""
    r = repr(e)
    if "out of memory" in r or "right_planner" in r or "left_planner" in r:
        print(f"setup poisoned ({r[:120]}), exiting", flush=True)
        sys.exit(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds_file", required=True)
    p.add_argument("--range", required=True, help="a:b, episodes a..b-1 of the seeds file")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n_exec", type=int, default=16)
    p.add_argument("--init_noise", type=float, default=-1.0)
    p.add_argument("--perception", default="")
    a = p.parse_args()
    lo, hi = (int(v) for v in a.range.split(":"))
    with open(a.seeds_file) as f:
        ref = json.load(f)
    seeds, instrs = ref["seeds"][lo:hi], ref["instructions"][lo:hi]
    assert len(seeds) == hi - lo, f"the seeds file has {len(ref['seeds'])} seeds, range {a.range}"
    out_dir = os.path.join(a.out, "parts", a.config)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{a.task}.{lo}-{hi}.json")
    name = os.path.basename(os.path.dirname(a.ckpt)) + (f"_k{a.k}" if a.k != 3 else "")
    res = {"task": a.task, "config": a.config, "ckpts": [a.ckpt], "seeds": [], "instructions": [], "results": {name: []},
           "range": [lo, hi], "done": False}
    if os.path.exists(out_path):                 # resume
        with open(out_path) as f:
            old = json.load(f)
        rep = {int(k): v for k, v in old.get("replaced", {}).items()}
        if old["ckpts"] == [a.ckpt] and old["seeds"] == [rep.get(s, s) for s in seeds[:len(old["seeds"])]]:
            res = old
    policy = SkeletonPolicy(a.ckpt, k=a.k, h=10, n_exec=a.n_exec, seed=0, consensus="medoid", flow_steps=0,
                            fixed_noise=False, temporal=0.0, grip_bin=False)       # evaluate.py's defaults
    percept = None
    if a.perception:
        from spikewam.robotwin.percept import RTFrontEnd
        assert not policy.model.cfg.motion_lag, "the policy would read true object motion between plans"
        percept = RTFrontEnd(a.perception, policy.device)
    task = make_task(a.task)
    args = make_args(a.task, a.config)
    args.update(render_freq=0, eval_mode=True, eval_video_log=False, policy_name="SpikeWAM", ckpt_setting="sw")
    for i in range(len(res["seeds"]), hi - lo):
        seed, instruction = seeds[i], instrs[i]
        # the scene's stability check at setup is not deterministic (a seed the reference accepted can come out
        # "unstable" here): retry the same seed. Tasks whose check_success reads what the expert run sets (arm_tag,
        # origin_z) get the expert run first, as in evaluate.py.
        for attempt in range(6):
            try:
                if a.task in EXPERT_FIRST:
                    task.setup_demo(now_ep_num=lo + i, seed=seed, is_test=True, **args)
                    task.play_once()
                    task.close_env()
                task.setup_demo(now_ep_num=lo + i, seed=seed, is_test=True, **args)
                break
            except Exception as e:
                print(f"{a.task} {a.config} seed {seed} setup attempt {attempt + 1}: {repr(e)[:200]}", flush=True)
                poisoned(e)
                task.close_env()
        else:
            # this seed's scene does not come out stable here (other runs skipped it too): replace it as evaluate.py
            # would, by the next new seed (above every seed of the file, a window per unit) that passes the expert check
            old_seed, seed = seed, max(ref["seeds"]) + 1 + 100 * lo
            used = set(ref["seeds"]) | set(res["seeds"])
            while True:
                if seed not in used:
                    try:
                        task.setup_demo(now_ep_num=lo + i, seed=seed, is_test=True, **args)
                        info = task.play_once()
                        ok, why = task.plan_success and task.check_success(), "expert plan failed or did not succeed"
                    except Exception as e:
                        ok, why = False, repr(e)[:200]
                        poisoned(e)
                    task.close_env()
                    if ok:
                        try:
                            task.setup_demo(now_ep_num=lo + i, seed=seed, is_test=True, **args)
                            break
                        except Exception as e:
                            why = repr(e)[:200]
                            task.close_env()
                    print(f"{a.task} {a.config} replacement seed {seed} skipped: {why}", flush=True)
                seed += 1
                if seed > max(ref["seeds"]) + 1 + 100 * lo + 300:
                    print(f"{a.task} {a.config}: 300 replacement seeds failed, giving up (the unit is retried)", flush=True)
                    sys.exit(3)
            random.seed(seed)
            desc = generate_episode_descriptions(a.task, [info["info"]], 100)
            instruction = str(np.random.choice(desc[0]["unseen"]))
            res.setdefault("replaced", {})[str(old_seed)] = seed
            print(f"{a.task} {a.config} seed {old_seed} replaced by {seed}: {instruction}", flush=True)
        task.set_instruction(instruction=instruction)
        succ, steps, plans, cm = rollout(task, policy, instruction, a.n_exec, None, a.init_noise, seed, percept)
        res["results"][name].append({"success": succ, "steps": steps, "plans": plans, **({"percept_cm": float(np.mean(cm)), "percept_cms": [round(float(c), 2) for c in cm]} if cm is not None else {})})
        task.close_env(clear_cache=((lo + i + 1) % args["clear_cache_freq"] == 0))
        res["seeds"].append(seed)
        res["instructions"].append(instruction)
        res["done"] = len(res["seeds"]) == hi - lo
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"{a.task} {a.config} seed {seed} ({lo + i + 1}, {i + 1}/{hi - lo} of {a.range}): success {succ}", flush=True)
    res["done"] = True
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
