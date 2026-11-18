"""Standard LIBERO evaluation with sim-state skeletons (no rendering): every task from its 50 benchmark initial
states, 10 settling steps with a no-op action, then closed-loop control with replanning every n_exec actions, up to the
suite's step limit (OpenVLA protocol).

  python -m spikewam.eval_libero --ckpt runs/r0_ann/ckpt_60000.pt --workers 24 [--suites ...] [--episodes 50]

Writes <ckpt dir>/eval_<ckpt stem>_<weights>[_k<k> | _tag].json (_k<k> when k != 3 and no tag is given) with
per-episode results and per-task / per-suite rates; --resume keeps the episodes already in that file and runs only
the missing ones (e.g. after workers died).
With --record, the skeleton of every step of every failed episode goes to traj_<same name>.pkl (float16) for the
failure-phase analysis. Every episode also records which goal predicates of the BDDL file hold at its end ("goals").
With --refine_every r, a fast loop refines the rest of the chunk every r actions between full plans
(SkeletonPolicy.refine: re-noised to --refine_tau, last flow steps on the current skeleton).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

from .libero_tasks import MAX_STEPS, NUM_STEPS_WAIT, SUITES, bddl_path, get_suite, make_env, scene_objects
from .skeleton import ROLES

DUMMY = [0, 0, 0, 0, 0, 0, -1]


def run_jobs(jobs, a, queue):
    import torch
    torch.set_num_threads(1)
    from .policy import SkeletonPolicy, retry_oom
    from .skeleton import SkeletonExtractor
    policy = retry_oom(SkeletonPolicy, a.ckpt, weights=a.weights, k=a.k, h=a.h, n_exec=a.n_exec,
                       seed=a.seed + jobs[0][2], consensus=a.consensus, flow_steps=getattr(a, "flow_steps", 0),
                       fixed_noise=getattr(a, "fixed_noise", False), temporal=getattr(a, "temporal", 0.0))
    skel_noise = getattr(a, "skel_noise", 0.0)
    rng = np.random.default_rng(a.seed + jobs[0][2])
    percept = None
    if getattr(a, "perception", ""):        # object nodes estimated from the two cameras, once per plan
        from .perception import FrontEnd
        assert not policy.model.cfg.motion_lag
        percept = FrontEnd(a.perception, policy.device)
        percept.smooth = getattr(a, "percept_smooth", 0)
        percept.hold = getattr(a, "percept_hold", False)
        mode = getattr(a, "percept_mode", "")
        percept.model.peak, percept.model.wta = "peak" in mode, "wta" in mode
    for suite_name, task_id, episodes in group_by_task(jobs):
        suite = get_suite(suite_name)
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = make_env(bddl_path(task), has_offscreen_renderer=percept is not None)
        for ep in episodes:
            env.reset()
            env.set_init_state(init_states[ep])
            names, ooi = scene_objects(env)
            ext = SkeletonExtractor(env.sim, names, ooi)
            policy.set_scene(ext, task.language)
            if percept is not None:
                percept.reset()
                front_text = percept.text_of(policy, task.language)
            success, plan, t, traj, since, percept_cm = False, [], 0, [], 0, []
            for t in range(MAX_STEPS[suite_name] + NUM_STEPS_WAIT):
                if t < NUM_STEPS_WAIT:
                    env.step(DUMMY)
                    continue
                skel = ext(env.sim)
                if percept is not None and not plan:
                    skel = percept(env.sim, ext, skel, front_text)
                    percept_cm.append(percept.last_cm)
                if skel_noise:      # perception error: object keypoints jitter, robot nodes (joint encoders) exact
                    obj = ext.role != ROLES["robot"]
                    skel = skel.copy()
                    skel[obj] += rng.normal(0.0, skel_noise, (int(obj.sum()), 3))
                policy.observe(skel)
                if a.record:
                    traj.append(skel.astype(np.float16))
                if not plan:
                    plan, since = list(policy.plan(skel)), 0
                elif a.refine_every and since % a.refine_every == 0:
                    plan = list(policy.refine(skel, since, a.refine_tau))
                _, _, done, _ = env.step(plan.pop(0).tolist())
                since += 1
                if done:
                    success = True
                    break
            inner = env.env
            rec = {"suite": suite_name, "task_id": task_id, "episode": ep, "success": success, "steps": t + 1,
                   "goals": [bool(inner._eval_predicate(g)) for g in inner.parsed_problem["goal_state"]]}
            if percept_cm:
                rec["percept_cm"] = float(np.mean(percept_cm))
            if a.record and not success:
                rec["traj"] = np.stack(traj)
                rec["role"], rec["slot"] = ext.role, ext.slot
            queue.put(rec)
        env.close()


def group_by_task(jobs):
    out = {}
    for s, t, ep in jobs:
        out.setdefault((s, t), []).append(ep)
    return [(s, t, eps) for (s, t), eps in out.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", default="ema")
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--tasks", default="", help="comma list of task ids (default all)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--consensus", default="medoid", choices=("medoid", "mean"),
                   help="how the k sampled chunks become one: MAC medoid or their average")
    p.add_argument("--flow_steps", type=int, default=0, help="Euler steps per sample (0: the checkpoint's)")
    p.add_argument("--fixed_noise", action="store_true",
                   help="one starting noise per episode, reused at every replanning step")
    p.add_argument("--temporal", type=float, default=0.0,
                   help="share of the previous plan kept for the steps both plans cover (leaky readout)")
    p.add_argument("--skel_noise", type=float, default=0.0,
                   help="std (m) of Gaussian noise added to every object keypoint at every step (perception error)")
    p.add_argument("--perception", default="", help="checkpoint of spikewam.perception: object nodes estimated from the cameras")
    p.add_argument("--percept_mode", default="", help="peak: decode at the distribution's peak; wta: when the views disagree, the more confident one")
    p.add_argument("--percept_smooth", type=int, default=0, help="median of this many plans for objects away from the hand")
    p.add_argument("--percept_hold", action="store_true", help="an object in the closed gripper is integrated in the hand frame")
    p.add_argument("--n_exec", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--refine_every", type=int, default=0, help="fast loop: refine the chunk every this many steps")
    p.add_argument("--refine_tau", type=float, default=0.3, help="fast loop: noise time the chunk is refined from")
    p.add_argument("--tag", default="")
    p.add_argument("--record", action="store_true", help="keep the skeleton trajectories of failed episodes")
    p.add_argument("--resume", action="store_true", help="run only the episodes missing from the output file")
    a = p.parse_args()
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    name = f"{stem}_{a.weights}{'_' + a.tag if a.tag else (f'_k{a.k}' if a.k != 3 else '')}"
    out = os.path.join(os.path.dirname(a.ckpt), f"eval_{name}.json")
    old = json.load(open(out))["episodes"] if a.resume and os.path.exists(out) else []
    done = {(r["suite"], r["task_id"], r["episode"]) for r in old}

    jobs = []
    for s in a.suites.split(","):
        ids = [int(x) for x in a.tasks.split(",")] if a.tasks else range(get_suite(s).n_tasks)
        jobs += [(s, t, ep) for t in ids for ep in range(a.episodes) if (s, t, ep) not in done]
    print(f"{len(jobs)} episodes to run ({len(old)} kept)", flush=True)
    # Contiguous blocks keep one environment per task inside a worker.
    chunks = [list(c) for c in np.array_split(np.array(jobs, dtype=object), a.workers) if len(c)]
    chunks = [[(s, int(t), int(e)) for s, t, e in c] for c in chunks]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=run_jobs, args=(c, a, queue)) for c in chunks]
    for pr in procs:
        pr.start()
    new, t0 = [], time.time()
    while len(new) < len(jobs):
        try:
            new.append(queue.get(timeout=60))
        except Exception:
            if not any(pr.is_alive() for pr in procs):
                break
            continue
        if len(new) % 100 == 0:
            rate = np.mean([r["success"] for r in new])
            print(f"{len(new)}/{len(jobs)} episodes, success {rate:.3f}, {time.time() - t0:.0f}s", flush=True)
    for pr in procs:
        pr.join()
    results = old + new

    summary = {"ckpt": a.ckpt, "weights": a.weights, "k": a.k, "h": a.h, "n_exec": a.n_exec,
               "consensus": a.consensus, "flow_steps": a.flow_steps, "fixed_noise": a.fixed_noise, "temporal": a.temporal,
               "skel_noise": a.skel_noise,
               "refine_every": a.refine_every, "refine_tau": a.refine_tau,
               "n_episodes": len(results), "suites": {}, "tasks": {}}
    for s in a.suites.split(","):
        rs = [r for r in results if r["suite"] == s]
        if rs:
            summary["suites"][s] = float(np.mean([r["success"] for r in rs]))
        for t in sorted({r["task_id"] for r in rs}):
            summary["tasks"][f"{s}/{t}"] = float(np.mean([r["success"] for r in rs if r["task_id"] == t]))
    summary["average"] = float(np.mean(list(summary["suites"].values()))) if summary["suites"] else 0.0
    if a.record:
        with open(os.path.join(os.path.dirname(a.ckpt), f"traj_{name}.pkl"), "wb") as f:
            pickle.dump([r for r in results if "traj" in r], f)
    summary["episodes"] = [{k: v for k, v in r.items() if k not in ("traj", "role", "slot")} for r in results]
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("suites", "average", "n_episodes")}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
