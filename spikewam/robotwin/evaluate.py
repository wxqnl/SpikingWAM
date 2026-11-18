"""Closed-loop RoboTwin 2.0 evaluation of skeleton policies (RoboTwin python with torch + transformers, cwd = the
RoboTwin root).

Follows RoboTwin's script/eval_policy.py: seeds from 100000 * (1 + --seed); a seed counts only if the expert solves
it (setup_demo + play_once, as RoboTwin's expert check); then the scene is reset with that seed, one instruction is
drawn from RoboTwin's episode descriptions of type --instruction_type (unseen by default), and the policy acts until
the task's step limit (task_config/_eval_step_limit.yml) or success; --test_num seeds per task. Two differences, neither
of which changes the simulation: every checkpoint given runs on each accepted seed with the same instruction (paired
comparison, one expert check), and no camera is rendered (the policies read the sim-state skeleton). Python's
`random` (RoboTwin's instruction generator) is seeded with the scene seed, so separate processes (one per checkpoint)
draw the same instruction for the same seed and stay paired.

Each plan samples a chunk from the skeleton and executes its first --n_exec joint targets with take_action (qpos).
--init_noise d: before the policy starts, the arm joints are moved from the home pose by N(0, d degrees) each (drawn
from the scene seed, the same for every checkpoint; the move does not count against the step limit): do the policies
bring a displaced arm back to the task? --init_noise 0 makes the same initial move without displacement (the control).
Each policy's sampling generator is reseeded from the scene seed at the start of every episode, so a seed sees the same
flow noise whatever ran before it.

  cd $RT_ROOT && $RT_PY $SW_ROOT/spikewam/robotwin/evaluate.py --task place_a2b_left --config demo_clean \
      --ckpt $SW_RUNS/rt_r0/ckpt_60000.pt $SW_RUNS/rt_f2/ckpt_60000.pt --out $SW_RUNS/robotwin_eval
Writes <out>/<config>/<task>.json (per checkpoint: successes, per-seed results). With --record, also
<out>/<config>/traj_<task>.pkl: per checkpoint and seed, at every plan the joint vector, the skeleton (float16) and the
executed joint targets, with the skeleton's roles, slots and object names (failure analysis).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "description", "utils"))
sys.path.insert(2, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from spikewam.policy import SkeletonPolicy  # noqa: E402
from spikewam.robotwin.rt_env import joints, make_args, make_task  # noqa: E402
from spikewam.robotwin.skeleton import RTSkeleton  # noqa: E402


ARM = [i for i in range(14) if i not in (6, 13)]


def rollout(task, policy, instruction, n_exec, rec=None, init_noise=-1.0, seed=0, percept=None):
    sk = RTSkeleton(task)
    policy.set_scene(sk, instruction)
    policy.gen.manual_seed(seed)
    plans, cms = 0, []
    if init_noise >= 0:
        q0 = joints(task)
        q0[ARM] += np.radians(np.random.default_rng(seed).normal(0, init_noise, len(ARM)))
        task.get_obs = lambda: None
        task.take_action(q0, action_type="qpos")
        task.take_action_cnt = 0
    skel = sk()
    policy.observe(skel)
    # take_action renders every camera once on success (get_obs, for the eval video); with many evaluators per GPU
    # that render can fail ("cannot create buffer") and kill a successful episode, so no observation is taken
    task.get_obs = lambda: None
    while task.take_action_cnt < task.step_lim and not task.eval_success:
        q = joints(task)                                        # the anchor of a posture-relative chunk
        seen = skel
        if percept is not None:                                 # object nodes from the cameras, once per plan
            seen = percept(task, sk, skel, policy.text_tokens)
            cms.append(percept.last_cm)
        chunk = policy.plan(seen, q)[:n_exec]
        if rec is not None:
            for k, v in (("q", q), ("skel", seen), ("act", chunk)):
                rec[k].append(np.asarray(v, np.float32 if k == "q" else np.float16))
        for a in chunk:
            task.take_action(a, action_type="qpos")
            skel = sk()
            policy.observe(skel)
            if task.eval_success or task.take_action_cnt >= task.step_lim:
                break
        plans += 1
    del task.get_obs
    if rec is not None:
        rec.update({k: np.stack(rec[k]) for k in ("q", "skel", "act")}, role=sk.role, slot=sk.slot,
                   objects=sk.object_names)
    return bool(task.eval_success), int(task.take_action_cnt), plans, (cms if cms else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", default="demo_clean", help="demo_clean (Easy) or demo_randomized (Hard)")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed_base", type=int, default=100000, help="seeds from seed_base * (1 + --seed); 100000 = RoboTwin's test seeds (diagnoses use others)")
    p.add_argument("--test_num", type=int, default=100)
    p.add_argument("--instruction_type", default="unseen")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--consensus", default="medoid", choices=("medoid", "mean"))
    p.add_argument("--flow_steps", type=int, default=0, help="Euler steps per sample (0: the checkpoint's)")
    p.add_argument("--fixed_noise", action="store_true", help="one starting noise per episode, reused at every replan")
    p.add_argument("--temporal", type=float, default=0.0, help="share of the previous plan kept (leaky readout)")
    p.add_argument("--grip_bin", action="store_true", help="grippers open or closed by the midpoint (as LIBERO's sign)")
    p.add_argument("--n_exec", type=int, default=16)
    p.add_argument("--record", action="store_true", help="keep every plan's joints, skeleton and joint targets")
    p.add_argument("--perception", default="", help="checkpoint of the spiking front end: object nodes estimated from the cameras")
    p.add_argument("--init_noise", type=float, default=-1.0,
                   help="degrees: displace the arm joints before the start (0: the same move, no displacement)")
    a = p.parse_args()
    out_dir = os.path.join(a.out, a.config)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{a.task}.json")
    infer = ((f"_k{a.k}" if a.k != 3 else "") + ("_mean" if a.consensus == "mean" else "")
             + (f"_steps{a.flow_steps}" if a.flow_steps else "") + ("_fixed" if a.fixed_noise else "")
             + (f"_tw{a.temporal:g}" if a.temporal else "") + ("_gb" if a.grip_bin else ""))
    names = [os.path.basename(os.path.dirname(c)) + infer for c in a.ckpt]
    res = {"task": a.task, "config": a.config, "ckpts": a.ckpt, "seeds": [], "instructions": [],
           "results": {n: [] for n in names}}
    if os.path.exists(out_path):                 # resume
        with open(out_path) as f:
            old = json.load(f)
        if old["ckpts"] == a.ckpt:
            res = old
    traj_path = os.path.join(out_dir, f"traj_{a.task}.pkl")
    traj = {n: {} for n in names}
    if a.record and os.path.exists(traj_path) and res["seeds"]:
        with open(traj_path, "rb") as f:
            traj = pickle.load(f)
    policies = [SkeletonPolicy(c, k=a.k, h=a.h, n_exec=a.n_exec, seed=a.seed, consensus=a.consensus,
                               flow_steps=a.flow_steps, fixed_noise=a.fixed_noise, temporal=a.temporal,
                               grip_bin=a.grip_bin) for c in a.ckpt]
    percept = None
    if a.perception:
        from spikewam.robotwin.percept import RTFrontEnd
        assert not any(p.model.cfg.motion_lag for p in policies), "the policy would read true object motion between plans"
        percept = RTFrontEnd(a.perception, policies[0].device)
    task = make_task(a.task)
    args = make_args(a.task, a.config)
    args.update(render_freq=0, eval_mode=True, eval_video_log=False, policy_name="SpikeWAM", ckpt_setting="sw")
    now_seed = a.seed_base * (1 + a.seed) if not res["seeds"] else res["seeds"][-1] + 1
    t0 = time.time()
    while len(res["seeds"]) < a.test_num:
        try:
            task.setup_demo(now_ep_num=len(res["seeds"]), seed=now_seed, is_test=True, **args)
            info = task.play_once()
            ok, why = task.plan_success and task.check_success(), "expert plan failed or did not succeed"
        except Exception as e:                   # e.g. CUDA out of memory in the planner: logged, never silent
            ok, why = False, repr(e)[:300]
        task.close_env()
        if not ok:
            print(f"{a.task} {a.config} seed {now_seed} skipped: {why}", flush=True)
            now_seed += 1
            continue
        instruction = None
        for name, policy in zip(names, policies):
            task.setup_demo(now_ep_num=len(res["seeds"]), seed=now_seed, is_test=True, **args)
            if instruction is None:
                random.seed(now_seed)
                desc = generate_episode_descriptions(a.task, [info["info"]], a.test_num)
                instruction = str(np.random.choice(desc[0][a.instruction_type]))
            task.set_instruction(instruction=instruction)
            rec = {"q": [], "skel": [], "act": []} if a.record else None
            succ, steps, plans, cm = rollout(task, policy, instruction, a.n_exec, rec, a.init_noise, now_seed, percept)
            if a.record:
                traj[name][now_seed] = dict(rec, success=succ)
            res["results"][name].append({"success": succ, "steps": steps, "plans": plans, **({"percept_cm": float(np.mean(cm)), "percept_cms": [round(float(c), 2) for c in cm]} if cm is not None else {})})
            task.close_env(clear_cache=((len(res["seeds"]) + 1) % args["clear_cache_freq"] == 0))
        res["seeds"].append(now_seed)
        res["instructions"].append(instruction)
        res["success"] = {n: float(np.mean([r["success"] for r in res["results"][n]])) for n in names}
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        if a.record:
            with open(traj_path, "wb") as f:
                pickle.dump(traj, f)
        print(f"{a.task} {a.config} seed {now_seed} ({len(res['seeds'])}/{a.test_num}, {time.time() - t0:.0f} s): "
              + "  ".join(f"{n} {res['success'][n] * 100:.1f}" for n in names), flush=True)
        now_seed += 1


if __name__ == "__main__":
    main()
