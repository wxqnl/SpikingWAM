"""Which part of RoboTwin's Hard randomization costs the front end its accuracy (no training, no policy rollout).

demo_randomized (Hard) differs from demo_clean (Easy) in four settings only: background textures (random_background,
clean_background_rate), clutter (cluttered_table), table height (random_table_height) and lighting (random_light,
crazy_random_light_rate). For every task and seed the scene is set up under Easy, Hard, and Hard with one of the four
turned back to its Easy value. As in evaluate.py a seed counts only if the expert solves it, the instruction is the
unseen one evaluate.py would draw, and the front end reads the cameras once at the start (the first plan, nothing
touched yet). Its object nodes, raw and after the rigid refit used in the closed loop, are stored with the true ones.

RoboTwin draws the table offset whether or not it is randomized and places the clutter after the task objects, so Hard
without table height or without clutter keeps Hard's random stream (same objects, textures and light): paired scene by
scene. The background and light switches change the stream (other placements), so those compare as distributions.

Seeds from --seed0 (60000: above every training round, below the test seeds 100000+).

  cd $RT_ROOT && CUDA_VISIBLE_DEVICES=2 $RT_PY $SW_ROOT/scripts/rt_factor_check.py --tasks handover_block,place_fan \
      --perception $SW_ROOT/runs/perception/rtB30tCM/ckpt.pt --out $SW_ROOT/runs/perception/checks/factor_rtB30tCM
Writes <out>/<variant>/<task>.npz.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

SW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(SW_ROOT, "spikewam", "robotwin"))

import numpy as np  # noqa: E402

from evaluate import make_args, make_task, SkeletonPolicy  # noqa: E402  (evaluate.py sets up the paths)
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from spikewam.robotwin.layout import N_ROBOT  # noqa: E402
from spikewam.robotwin.percept import RTFrontEnd  # noqa: E402
from spikewam.robotwin.skeleton import RTSkeleton  # noqa: E402

EASY = {"random_background": False, "clean_background_rate": 1, "cluttered_table": False, "random_table_height": 0,
        "random_light": False, "crazy_random_light_rate": 0}
FACTORS = {"background": ("random_background", "clean_background_rate"), "clutter": ("cluttered_table",),
           "table": ("random_table_height",), "light": ("random_light", "crazy_random_light_rate")}
VARIANTS = ["easy", "hard", "hard_no_table", "hard_no_clutter", "hard_no_background", "hard_no_light"]


def variant_args(task_name, variant):
    args = make_args(task_name, "demo_randomized")
    dr = args["domain_randomization"]
    if variant == "easy":
        dr.update(EASY)
    elif variant.startswith("hard_no_"):
        for k in FACTORS[variant[len("hard_no_"):]]:
            dr[k] = EASY[k]
    args.update(render_freq=0, eval_mode=True, eval_video_log=False, policy_name="SpikeWAM", ckpt_setting="sw")
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="comma list")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--perception", required=True)
    p.add_argument("--ckpt", default=os.path.join(SW_ROOT, "runs/rt_f2ppt16_big/ckpt_150000.pt"), help="policy: its text encoder only")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4, help="accepted seeds per task and variant")
    p.add_argument("--seed0", type=int, default=60000)
    p.add_argument("--max_tries", type=int, default=40)
    a = p.parse_args()
    policy = SkeletonPolicy(a.ckpt, k=3, h=10, n_exec=16, seed=0)
    percept = RTFrontEnd(a.perception, policy.device)
    for task_name in a.tasks.split(","):
        task = make_task(task_name)
        for variant in a.variants.split(","):
            out_path = os.path.join(a.out, variant, f"{task_name}.npz")
            if os.path.exists(out_path):
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            args = variant_args(task_name, variant)
            rec = {k: [] for k in ("seed", "true", "est", "raw", "head", "table_z_bias", "crazy_light", "instruction", "role", "slot", "names")}
            seed, t0 = a.seed0, time.time()
            while len(rec["seed"]) < a.n and seed < a.seed0 + a.max_tries:
                try:
                    task.setup_demo(now_ep_num=len(rec["seed"]), seed=seed, is_test=True, **copy.deepcopy(args))
                    info = task.play_once()
                    ok, why = task.plan_success and task.check_success(), "expert plan failed or did not succeed"
                except Exception as e:
                    ok, why = False, repr(e)[:200]
                task.close_env()
                if not ok:
                    print(f"{task_name} {variant} seed {seed} skipped: {why}", flush=True)
                    seed += 1
                    continue
                try:
                    task.setup_demo(now_ep_num=len(rec["seed"]), seed=seed, is_test=True, **copy.deepcopy(args))
                except Exception as e:
                    print(f"{task_name} {variant} seed {seed} second setup failed: {repr(e)[:200]}", flush=True)
                    task.close_env()
                    seed += 1
                    continue
                random.seed(seed)
                desc = generate_episode_descriptions(task_name, [info["info"]], 100)
                instruction = str(np.random.choice(desc[0]["unseen"]))
                task.set_instruction(instruction=instruction)
                sk = RTSkeleton(task)
                policy.set_scene(sk, instruction)
                skel = sk()
                percept.rigid = False
                raw = percept(task, sk, skel, policy.text_tokens)
                percept.rigid = True
                est = percept(task, sk, skel, policy.text_tokens)
                rec["seed"].append(seed)
                rec["true"].append(skel[N_ROBOT:].astype(np.float32))
                rec["est"].append(est[N_ROBOT:].astype(np.float32))
                rec["raw"].append(raw[N_ROBOT:].astype(np.float32))
                rec["head"].append(np.asarray(percept.last_views[1][0], np.float32))
                rec["table_z_bias"].append(float(task.table_z_bias))
                rec["crazy_light"].append(bool(task.crazy_random_light))
                rec["instruction"].append(instruction)
                rec["role"].append(np.asarray(sk.role[N_ROBOT:]))
                rec["slot"].append(np.asarray(sk.slot[N_ROBOT:]))
                rec["names"].append(np.array(list(sk.object_names)))
                task.close_env(clear_cache=True)
                print(f"{task_name} {variant} seed {seed}: {np.linalg.norm(est[N_ROBOT:] - skel[N_ROBOT:], axis=-1).mean() * 100:.2f} cm "
                      f"(raw {np.linalg.norm(raw[N_ROBOT:] - skel[N_ROBOT:], axis=-1).mean() * 100:.2f}), {time.time() - t0:.0f} s", flush=True)
                seed += 1
            if not rec["seed"]:
                print(f"{task_name} {variant}: no accepted seed", flush=True)
                continue
            np.savez(out_path, seed=np.array(rec["seed"]), true=np.array(rec["true"], dtype=object), est=np.array(rec["est"], dtype=object),
                     raw=np.array(rec["raw"], dtype=object), head=np.stack(rec["head"]), table_z_bias=np.array(rec["table_z_bias"]),
                     crazy_light=np.array(rec["crazy_light"]), instruction=np.array(rec["instruction"]),
                     role=np.array(rec["role"], dtype=object), slot=np.array(rec["slot"], dtype=object), object_names=np.array(rec["names"], dtype=object))
        print(f"{task_name} done", flush=True)


if __name__ == "__main__":
    main()
