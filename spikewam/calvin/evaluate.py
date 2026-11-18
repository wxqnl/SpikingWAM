"""CALVIN long-horizon evaluation with sim-state skeletons (no rendering), the official protocol
(spikewam/calvin/protocol.py): the 1,000 chains of data/calvin/eval_sequences.json in the environment of the split's
validation configuration (scene D for ABC -> D), each from its initial state, 5 subtasks in a row, 360 steps per
subtask, the instruction = the first validation annotation of the subtask, success by the CALVIN task oracle against
the state at the subtask's start; a chain stops at its first failed subtask. Replanning every n_exec actions.

  python -m spikewam.calvin.evaluate --ckpt runs/<run>/ckpt_30000.pt --gpus 0,5,6,7 --workers 64 [--n 1000]

Writes <ckpt dir>/calvin_<ckpt stem>_<weights>[_tag].json: per-chain results, the success rate of chain lengths 1-5,
the average length and per-task rates; --resume runs only the chains missing from that file.
Diagnostics (not the protocol): --scene runs the chains in another scene (e.g. calvin_scene_A, a training scene whose
sliding door sits where D's does), --instr train gives each subtask the most frequent training instruction of its task
instead of the unseen validation one.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np

from .protocol import CALVIN_ROOT, EP_LEN

SPLIT = "/mnt/data/spoil/workspace/datasets/calvin/task_ABC_D/validation"


def run_jobs(jobs, a, gpu, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import hydra
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(1)
    from ..policy import SkeletonPolicy, retry_oom
    from .skeleton import CalvinSkeleton, make_env
    conf = os.path.join(CALVIN_ROOT, "calvin_models", "conf")
    oracle = hydra.utils.instantiate(OmegaConf.load(os.path.join(conf, "callbacks/rollout/tasks/new_playtable_tasks.yaml")))
    annotations = OmegaConf.load(os.path.join(conf, "annotations/new_playtable_validation.yaml"))
    seqs = json.load(open(a.sequences))
    instr = {t: annotations[t][0] for t in annotations}
    if getattr(a, "instr", "val") == "train":
        import collections
        train = collections.defaultdict(collections.Counter)
        for f in glob.glob(os.path.join(os.environ.get("SW_DATA", "data"), "calvin", "task_ABC_D", "training", "*.npz")):
            with np.load(f) as z:
                for t, l in zip(z["task"], z["language"]):
                    train[str(t)][str(l)] += 1
        instr = {t: train[t].most_common(1)[0][0] for t in instr}
    policy = retry_oom(SkeletonPolicy, a.ckpt, weights=a.weights, k=a.k, h=a.h, n_exec=a.n_exec, seed=a.seed + jobs[0],
                       consensus=a.consensus, flow_steps=a.flow_steps, fixed_noise=a.fixed_noise)
    scene = getattr(a, "scene", "") or None
    env = make_env(a.split, scene)
    sk = CalvinSkeleton(make_env(a.split, scene))       # kinematic copy of the same scene
    for i in jobs:
        s = seqs[i]
        obs = env.reset(robot_obs=np.array(s["robot_obs"]), scene_obs=np.array(s["scene_obs"]))
        subtasks = []
        for task in s["sequence"]:
            policy.set_scene(sk, instr[task])
            start_info = env.get_info()
            plan, success, t = [], False, 0
            for t in range(EP_LEN):
                skel = sk(obs["robot_obs"], obs["scene_obs"])
                policy.observe(skel)
                if not plan:
                    plan = list(policy.plan(skel))
                obs, _, _, info = env.step(plan.pop(0).copy())
                if oracle.get_task_info_for_set(start_info, info, {task}):
                    success = True
                    break
            subtasks.append({"task": task, "success": success, "steps": t + 1})
            if not success:
                break
        queue.put({"seq": i, "sequence": s["sequence"], "n_done": sum(x["success"] for x in subtasks),
                   "subtasks": subtasks})


def summarize(results):
    n = np.array([r["n_done"] for r in results])
    tasks = {}
    for r in results:
        for x in r["subtasks"]:
            tasks.setdefault(x["task"], []).append(x["success"])
    return {"n_chains": len(results), "success_at": [float(np.mean(n >= k)) for k in range(1, 6)],
            "avg_len": float(n.mean()), "tasks": {t: [float(np.mean(v)), len(v)] for t, v in sorted(tasks.items())}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", default="ema")
    p.add_argument("--sequences", default=os.path.join(os.environ.get("SW_DATA", "data"), "calvin",
                                                       "eval_sequences.json"))
    p.add_argument("--split", default=SPLIT, help="validation split whose configuration builds the environment")
    p.add_argument("--n", type=int, default=1000, help="the first n chains")
    p.add_argument("--gpus", default="0")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--consensus", default="medoid", choices=("medoid", "mean"))
    p.add_argument("--flow_steps", type=int, default=1, help="Euler steps per sample (0: the checkpoint's)")
    p.add_argument("--fixed_noise", type=int, default=1, help="one starting noise per subtask (persistent stream)")
    p.add_argument("--n_exec", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scene", default="", help="diagnostic: run the chains in this scene instead of the split's")
    p.add_argument("--instr", default="val", choices=("val", "train"), help="diagnostic: training instructions")
    p.add_argument("--tag", default="")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.fixed_noise = bool(a.fixed_noise)
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    out = os.path.join(os.path.dirname(a.ckpt), f"calvin_{stem}_{a.weights}{'_' + a.tag if a.tag else ''}.json")
    old = json.load(open(out))["chains"] if a.resume and os.path.exists(out) else []
    done = {r["seq"] for r in old}
    jobs = [i for i in range(a.n) if i not in done]
    print(f"{len(jobs)} chains to run ({len(old)} kept)", flush=True)
    gpus = a.gpus.split(",")
    chunks = [list(map(int, c)) for c in np.array_split(np.array(jobs), a.workers) if len(c)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=run_jobs, args=(c, a, gpus[w % len(gpus)], queue)) for w, c in enumerate(chunks)]
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
        if len(new) % 50 == 0:
            s = summarize(new)
            print(f"{len(new)}/{len(jobs)} chains, " + " ".join(f"{x:.3f}" for x in s["success_at"]) +
                  f", avg len {s['avg_len']:.3f}, {time.time() - t0:.0f}s", flush=True)
    for pr in procs:
        pr.join()
    results = sorted(old + new, key=lambda r: r["seq"])
    summary = {"ckpt": a.ckpt, "weights": a.weights, "k": a.k, "h": a.h, "n_exec": a.n_exec, "consensus": a.consensus,
               "flow_steps": a.flow_steps, "fixed_noise": a.fixed_noise, "split": a.split, "seed": a.seed,
               "scene": a.scene, "instr": a.instr}
    summary |= summarize(results) | {"chains": results}
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("n_chains", "success_at", "avg_len")}))
    print("wrote", out)


if __name__ == "__main__":
    main()
