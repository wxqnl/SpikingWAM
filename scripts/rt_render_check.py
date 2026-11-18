"""Is the first camera render after a RoboTwin scene setup different from the next ones? (no training, no rollout)

The closed loop (evaluate.py) estimates the first plan's object nodes from the first render after setup_demo; the
factor check (rt_factor_check.py) called the front end twice on the same state and kept the second render's estimate.
On Hard the closed loop's first-plan error is about twice the factor check's, on Easy the same. Here the episode is set
up exactly as evaluate.py does (expert check, reset, unseen instruction), then the front end (rigid refit) reads the
cameras N times without any step in between: per call the mean object-node error and the mean absolute image change
from the previous render, per camera. Seeds from --seed0 (70000: not the test seeds).

  cd $RT_ROOT && CUDA_VISIBLE_DEVICES=3 $RT_PY $SW_ROOT/scripts/rt_render_check.py --tasks handover_block,place_mouse_pad \
      --perception $SW_ROOT/runs/perception/rtB30tCM/ckpt.pt --config demo_randomized
"""

from __future__ import annotations

import argparse
import os
import random
import sys

SW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(SW_ROOT, "spikewam", "robotwin"))

import numpy as np  # noqa: E402

from evaluate import make_args, make_task, SkeletonPolicy  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from spikewam.robotwin.layout import N_ROBOT  # noqa: E402
from spikewam.robotwin.percept import RTFrontEnd  # noqa: E402
from spikewam.robotwin.skeleton import RTSkeleton  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--config", default="demo_randomized")
    p.add_argument("--perception", required=True)
    p.add_argument("--ckpt", default=os.path.join(SW_ROOT, "runs/rt_f2ppt16_big/ckpt_150000.pt"))
    p.add_argument("--n", type=int, default=3, help="accepted seeds per task")
    p.add_argument("--calls", type=int, default=4)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--seen_textures", action="store_true", help="eval_mode off: backgrounds from RoboTwin's seen (training) set")
    a = p.parse_args()
    policy = SkeletonPolicy(a.ckpt, k=3, h=10, n_exec=16, seed=0)
    percept = RTFrontEnd(a.perception, policy.device)
    percept.rigid = True
    for task_name in a.tasks.split(","):
        task = make_task(task_name)
        args = make_args(task_name, a.config)
        args.update(render_freq=0, eval_mode=not a.seen_textures, eval_video_log=False, policy_name="SpikeWAM", ckpt_setting="sw")
        seed, done = a.seed0, 0
        while done < a.n and seed < a.seed0 + 30:
            try:
                task.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args)
                info = task.play_once()
                ok = task.plan_success and task.check_success()
            except Exception:
                ok = False
            task.close_env()
            if not ok:
                seed += 1
                continue
            try:
                task.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args)
            except Exception as e:
                print(f"{task_name} seed {seed} setup failed: {repr(e)[:120]}", flush=True)
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
            prev, cells = None, []
            for c in range(a.calls):
                est = percept(task, sk, skel, policy.text_tokens)
                imgs = np.asarray(percept.last_views[0], np.float32)
                err = np.linalg.norm(est[N_ROBOT:] - skel[N_ROBOT:], axis=-1).mean() * 100
                diff = "" if prev is None else " img change " + "/".join(f"{np.abs(imgs[v] - prev[v]).mean():.2f}" for v in range(len(imgs)))
                cells.append(f"call {c}: {err:.2f} cm{diff}")
                prev = imgs
            print(f"{task_name} {a.config} {'seen' if a.seen_textures else 'unseen'} seed {seed} light {getattr(task, 'crazy_random_light', None)} "
                  f"wall {getattr(task, 'wall_texture', None)} table {getattr(task, 'table_texture', None)} tz {float(task.table_z_bias) * 100:.2f}: " + " | ".join(cells), flush=True)
            task.close_env(clear_cache=True)
            done += 1
            seed += 1


if __name__ == "__main__":
    main()
