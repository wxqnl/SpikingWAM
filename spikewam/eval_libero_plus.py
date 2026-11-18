"""LIBERO-Plus (arXiv 2510.13626) with sim-state skeletons: every perturbed task once, from its own initial state,
with the LIBERO protocol of eval_libero (10 settling steps, the suite's step limit, replanning every n_exec actions).

Run under setup/libero_plus_env.sh (LIBERO-Plus on the path instead of LIBERO). Instructions come from Spike-VLA's
libero_plus_instructions.json: LIBERO-Plus builds them from task names, which carry the perturbation suffix for every
task without `_language_` ("... on the plate light 10"). Sim-state skeletons read no image, so the camera, light,
background and sensor-noise categories only change what the robot sees through the scene itself (e.g. none); the
sensor noise, which LIBERO-Plus applies to the agent-view image in step(), is switched off.

  python -m spikewam.eval_libero_plus --ckpt runs/<run>/ckpt_60000.pt --workers 32 [--categories ...]

Results go to <ckpt dir>/libero_plus_<ckpt stem>_<weights>/{episodes.jsonl, summary.json}; a rerun skips the tasks
already in episodes.jsonl.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time

import numpy as np

from .libero_tasks import MAX_STEPS, NUM_STEPS_WAIT, bddl_path, get_suite, make_env, scene_objects

PLUS_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DUMMY = [0, 0, 0, 0, 0, 0, -1]
MAX_RESET_TRIES = 100


def bounded_reset(self):
    from robosuite.utils.errors import RandomizationError
    for _ in range(MAX_RESET_TRIES):
        try:
            return self.env.reset()
        except RandomizationError:
            pass
    raise RuntimeError(f"object placement failed {MAX_RESET_TRIES} times")


def corrupt_agentview(image, noise):
    """The sensor noise LIBERO-Plus applies to the third-person image in step()."""
    from libero.libero.envs import env_wrapper as w
    from PIL import Image
    pil = Image.fromarray(image)
    if noise <= 10:
        out = w.motion_blur(pil, severity=noise)
    elif noise <= 20:
        out = w.gaussian_blur(pil, severity=noise - 10)
    elif noise <= 30:
        out = w.zoom_blur(pil, severity=noise - 20)
    elif noise <= 40:
        out = w.fog(pil, severity=noise - 30)
    else:
        out = w.glass_blur(pil, severity=noise - 40)
    return np.asarray(out).astype(np.uint8)


def run_chunk(args):
    chunk, a, instructions = args
    import torch
    torch.set_num_threads(1)
    from libero.libero.envs.env_wrapper import ControlEnv
    ControlEnv.reset = bounded_reset
    from .policy import SkeletonPolicy, retry_oom
    from .skeleton import SkeletonExtractor
    policy = retry_oom(SkeletonPolicy, a.ckpt, weights=a.weights, k=a.k, h=a.h, n_exec=a.n_exec,
                       seed=a.seed + chunk[0][1], consensus=a.consensus, flow_steps=getattr(a, "flow_steps", 0),
                       fixed_noise=getattr(a, "fixed_noise", False), temporal=getattr(a, "temporal", 0.0))
    percept = None
    if a.perception:                 # object nodes estimated from the two cameras, once per plan
        from .perception import FrontEnd
        percept = FrontEnd(a.perception, policy.device)
        percept.smooth = a.percept_smooth
        percept.trace = a.percept_trace
        percept.hold = getattr(a, "percept_hold", False)
        percept.model.peak, percept.model.wta = "peak" in a.percept_mode, "wta" in a.percept_mode
    out = []
    for suite_name, idx, category in chunk:
        suite = get_suite(suite_name)
        task = suite.get_task(idx)
        rec = {"suite": suite_name, "task_index": idx, "task_name": task.name, "category": category}
        try:
            env = make_env(bddl_path(task), has_offscreen_renderer=percept is not None)
            sensor_noise, env.noise = env.noise, 0
            if percept is not None:
                percept.corrupt = (lambda im, n=sensor_noise: corrupt_agentview(im, n)) if sensor_noise else None
            env.seed(a.seed)
            env.reset()
            env.set_init_state(suite.get_task_init_states(idx)[0])
            names, ooi = scene_objects(env)
            ext = SkeletonExtractor(env.sim, names, ooi)
            policy.set_scene(ext, instructions[task.name])
            front_text = percept.text_of(policy, instructions[task.name]) if percept is not None else None
            if getattr(a, "percept_text", "") == "base" and "_language_" in task.name:     # diagnosis: the front end reads the task's own sentence, the policy the reworded one
                sentence = re.sub(r"_language_\d+.*$", "", task.name)
                sentence = re.sub(r"^[A-Z_]*SCENE\d+_", "", sentence).replace("_", " ")
                front_text = policy.text([sentence])[0]
            if percept is not None:
                percept.reset()
            success, plan, t, percept_cm, trace = False, [], 0, [], []
            for t in range(MAX_STEPS[suite_name] + NUM_STEPS_WAIT):
                if t < NUM_STEPS_WAIT:
                    env.step(DUMMY)
                    continue
                skel = ext(env.sim)
                if percept is not None and not plan:
                    skel = percept(env.sim, ext, skel, front_text)
                    percept_cm.append(percept.last_cm)
                    if percept.trace:
                        trace.append({"t": t, **percept.last_detail})
                policy.observe(skel)
                if not plan:
                    plan = list(policy.plan(skel))
                _, _, done, _ = env.step(plan.pop(0).tolist())
                if done:
                    success = True
                    break
            env.close()
            rec |= {"success": success, "steps": t + 1}
            if percept_cm:
                rec["percept_cm"] = float(np.mean(percept_cm))
            if trace:
                rec |= {"trace": trace, "slot": np.asarray(ext.slot)[-len(trace[0]["fused"]):].tolist(), "kinds": ext.kinds}
        except Exception as exc:  # placement failures and broken scenes count as failures, recorded as such
            rec |= {"success": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", default="ema")
    p.add_argument("--suites", default=",".join(PLUS_SUITES))
    p.add_argument("--categories", default="all", help="comma list of LIBERO-Plus categories, or all")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--consensus", default="medoid", choices=("medoid", "mean"),
                   help="how the k sampled chunks become one: MAC medoid or their average")
    p.add_argument("--flow_steps", type=int, default=0, help="Euler steps per sample (0: the checkpoint's)")
    p.add_argument("--fixed_noise", action="store_true",
                   help="one starting noise per episode, reused at every replanning step")
    p.add_argument("--temporal", type=float, default=0.0,
                   help="share of the previous plan kept for the steps both plans cover (leaky readout)")
    p.add_argument("--n_exec", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--instructions", default=os.path.join(os.environ.get("SW_BENCH", ""),
                                                           "libero_plus_instructions.json"))
    p.add_argument("--perception", default="", help="checkpoint of spikewam.perception: object nodes estimated from the cameras")
    p.add_argument("--percept_mode", default="", help="peak: decode at the distribution's peak; wta: when the views disagree, the more confident one")
    p.add_argument("--percept_trace", action="store_true", help="record per-node errors of each view at every plan")
    p.add_argument("--percept_smooth", type=int, default=0, help="median of this many plans for objects away from the hand")
    p.add_argument("--percept_hold", action="store_true", help="an object in the closed gripper is integrated in the hand frame")
    p.add_argument("--percept_text", default="", help="base: for reworded tasks the front end reads the task's own sentence (diagnosis, not used for reported numbers)")
    p.add_argument("--tag", default="", help="suffix of the output directory")
    p.add_argument("--limit_per_category", type=int, default=0, help="evenly spaced tasks per category (a sample of the benchmark)")
    a = p.parse_args()

    from libero.libero import get_libero_path
    table = json.load(open(os.path.join(get_libero_path("benchmark_root"), "benchmark", "task_classification.json")))
    instructions = json.load(open(a.instructions))["instructions"]
    cats = None if a.categories == "all" else set(a.categories.split(","))
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    out_dir = os.path.join(os.path.dirname(a.ckpt),
                           f"libero_plus_{stem}_{a.weights}" + (f"_k{a.k}" if a.k != 3 else "")
                           + ("_mean" if a.consensus == "mean" else "")
                           + (f"_steps{a.flow_steps}" if a.flow_steps else "") + ("_fixed" if a.fixed_noise else "") + (f"_tw{a.temporal:g}" if a.temporal else "") + (f"_{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    ep_path = os.path.join(out_dir, "episodes.jsonl")
    done = set()
    if os.path.exists(ep_path):
        done = {(r["suite"], r["task_index"]) for r in map(json.loads, open(ep_path))}
    jobs = [(s, row["id"] - 1, row["category"]) for s in a.suites.split(",") for row in table[s]
            if cats is None or row["category"] in cats]
    if a.limit_per_category:
        sample = []
        for c in sorted({j[2] for j in jobs}):
            rows = [j for j in jobs if j[2] == c]
            sample += rows[::max(1, len(rows) // a.limit_per_category)][:a.limit_per_category]
        jobs = sample
    jobs = [j for j in jobs if (j[0], j[1]) not in done]   # after sampling: a resumed sample is the same sample
    chunks =[jobs[i:i + a.chunk] for i in range(0, len(jobs), a.chunk)]
    print(f"{len(jobs)} tasks to run ({len(done)} already done) in {len(chunks)} chunks", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(ep_path, "a") as f:
        for recs in pool.imap_unordered(run_chunk, [(c, a, instructions) for c in chunks]):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += len(recs)
            print(f"{n}/{len(jobs)} tasks, {time.time() - t0:.0f}s", flush=True)

    recs = [json.loads(l) for l in open(ep_path)]
    summary = {"ckpt": a.ckpt, "weights": a.weights, "n_tasks": len(recs),
               "n_errors": sum("error" in r for r in recs), "categories": {}, "suites": {}}
    for c in sorted({r["category"] for r in recs}):
        summary["categories"][c] = float(np.mean([r["success"] for r in recs if r["category"] == c]))
    for s in sorted({r["suite"] for r in recs}):
        summary["suites"][s] = float(np.mean([r["success"] for r in recs if r["suite"] == s]))
    summary["overall"] = float(np.mean([r["success"] for r in recs]))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
