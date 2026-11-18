"""Front-end images from the policy's own closed loop (RoboTwin python, cwd = the RoboTwin root).

The policy acts on the front end's estimates as in evaluation (spikewam.robotwin.evaluate.rollout), and every plan stores
the three views, their world-to-pixel matrices and the true skeleton, in the format of perception_cache.py, so the front
end can be trained on the states the policy itself reaches (successes and failures alike). Nothing of the evaluation
enters: training seeds (--seed0, far below the evaluation seeds 100000 and up), seen instructions, and the training
textures (eval_mode off; RoboTwin's eval mode switches the randomized scenes to unseen textures).

  cd $RT_ROOT && PERCEPT_RIGID=1 $RT_PY $SW_ROOT/spikewam/robotwin/perception_cl.py --task place_shoe --config demo_clean \
      --ckpt $SW_ROOT/runs/rt_f2ppt16_big/ckpt_150000.pt --perception $SW_ROOT/runs/perception/rt160h/ckpt.pt --out $OUT
Writes <out>/<config>_<task>/{rgb.npy, meta.npz}.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "description", "utils"))
sys.path.insert(2, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from spikewam.policy import SkeletonPolicy  # noqa: E402
from spikewam.robotwin.evaluate import rollout  # noqa: E402
from spikewam.robotwin.perception_cache import N_TEXT, SIZE, views  # noqa: E402
from spikewam.robotwin.percept import RTFrontEnd  # noqa: E402
from spikewam.robotwin.rt_env import make_args, make_task  # noqa: E402
from spikewam.robotwin.skeleton import RTSkeleton  # noqa: E402


class Recorder:
    """Wraps the front end: passes its estimates on and keeps the views it read with the true skeleton (t = control step)."""

    def __init__(self, front):
        self.front, self.frames, self.sk = front, [], None

    def __call__(self, task, sk, skel, text):
        est = self.front(task, sk, skel, text)
        images, mats = self.front.last_views
        self.sk = sk
        self.frames.append((int(task.take_action_cnt), images, mats, np.array(skel, np.float32)))
        return est

    def hook(self, task, every, n_exec):
        """Also keep the views every `every` control steps inside a chunk (the plan steps are kept by __call__)."""
        act = task.take_action

        def take_action(action, action_type="qpos"):
            out = act(action, action_type=action_type)
            t = int(task.take_action_cnt)
            if self.sk is not None and t % every == 0 and t % n_exec and not task.eval_success:
                images, mats = views(task)
                self.frames.append((t, images, mats, np.array(self.sk(), np.float32)))
            return out

        task.take_action = take_action

    @property
    def last_cm(self):
        return self.front.last_cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed0", type=int, default=5000)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--perception", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=7200, help="seconds; stop collecting and save what there is")
    p.add_argument("--retry", action="store_true")
    p.add_argument("--every", type=int, default=4, help="also keep the views every this many control steps inside a chunk (0: plan steps only)")
    a = p.parse_args()
    d = os.path.join(a.out, f"{a.config}_{a.task}")
    if os.path.exists(os.path.join(d, "meta.npz")):
        print("done already", d, flush=True)
        return
    try:
        os.makedirs(d, exist_ok=a.retry)
    except FileExistsError:
        print("taken by another process", d, flush=True)
        return
    with open("./task_config/_eval_step_limit.yml") as f:
        step_lim = yaml.safe_load(f)[a.task]
    policy = SkeletonPolicy(a.ckpt, k=3, h=10, n_exec=16, seed=0, consensus="medoid", flow_steps=0, fixed_noise=False, temporal=0.0, grip_bin=False)
    rec = Recorder(RTFrontEnd(a.perception, policy.device))
    task = make_task(a.task)
    args = make_args(a.task, a.config)
    args.update(render_freq=0, eval_video_log=False, policy_name="SpikeWAM", ckpt_setting="sw")
    eps, seed, wins, t0, errors = [], a.seed0, 0, time.time(), 0
    while len(eps) < a.episodes and time.time() - t0 < a.budget and seed < a.seed0 + 2000:     # far below the evaluation seeds
        try:
            task.setup_demo(now_ep_num=len(eps), seed=seed, **args)
            info = task.play_once()
            ok = task.plan_success and task.check_success()
            errors = 0
        except Exception as e:
            ok = False
            errors += 1
            print(f"seed {seed}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        if errors >= 20:                      # a broken process (e.g. no motion planner after an out-of-memory start)
            break
        task.close_env()
        if not ok:
            seed += 1
            continue
        task.setup_demo(now_ep_num=len(eps), seed=seed, **args)
        task.step_lim = step_lim
        random.seed(seed)
        seen = list(generate_episode_descriptions(a.task, [info["info"]], 100)[0]["seen"])
        instruction = str(np.random.default_rng(seed).choice(seen))
        task.set_instruction(instruction=instruction)
        rec.frames, rec.sk = [], None
        if a.every:
            rec.hook(task, a.every, 16)
        succ, steps, plans, cms = rollout(task, policy, instruction, 16, None, -1.0, seed, rec)
        if a.every:
            del task.take_action
        sk = RTSkeleton(task)
        kinds = ["robot"] * sk.n_nodes
        for i, s in enumerate(sk.slot):
            if s:
                kinds[i] = re.sub(r"_\d+$", "", sk.object_names[s - 1])
        text = [instruction] + [seen[i % len(seen)] for i in range(N_TEXT - 1)]
        eps.append({"seed": seed, "frames": rec.frames, "node_id": sk.node_id, "role": sk.role, "slot": sk.slot, "kinds": kinds, "text": text, "success": bool(succ)})
        wins += succ
        print(f"episode {len(eps)}: seed {seed}, success {succ}, {len(rec.frames)} frames, front end {np.mean(cms):.1f} cm ({time.time() - t0:.0f} s)", flush=True)
        task.close_env(clear_cache=((seed + 1) % args["clear_cache_freq"] == 0))
        seed += 1
    if not eps:                               # nothing kept: give the task back so a later pass can take it
        os.rmdir(d)
        raise SystemExit(f"{a.config} {a.task}: no episode ({errors} errors in a row, last seed {seed})")
    n_frames = sum(len(e["frames"]) for e in eps)
    n_pad = max(len(e["node_id"]) for e in eps)
    rgb = np.lib.format.open_memmap(os.path.join(d, "rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(n_frames, 3, SIZE, SIZE, 3))
    skel = np.zeros((n_frames, n_pad, 3), np.float32)
    desc = {k: np.zeros((n_frames, n_pad), np.int64) for k in ("node_id", "role", "slot")}
    M, ep_i, t_i, n_nodes = np.zeros((n_frames, 3, 3, 4), np.float32), np.zeros(n_frames, np.int64), np.zeros(n_frames, np.int64), np.zeros(n_frames, np.int64)
    vocab = sorted({k for e in eps for k in e["kinds"]})
    index = {k: i for i, k in enumerate(vocab)}
    kind = np.zeros((n_frames, n_pad), np.int64)
    i = 0
    for k, e in enumerate(eps):
        n = len(e["node_id"])
        for t, images, mats, s in e["frames"]:
            rgb[i], M[i], skel[i, :n], ep_i[i], t_i[i], n_nodes[i] = images, mats, s, k, t, n
            for name in desc:
                desc[name][i, :n] = e[name]
            kind[i, :n] = [index[x] for x in e["kinds"]]
            i += 1
    rgb.flush()
    del rgb
    os.replace(os.path.join(d, "rgb.tmp.npy"), os.path.join(d, "rgb.npy"))
    np.savez(os.path.join(d, "meta.npz"), skel=skel, M=M, ep=ep_i, t=t_i, n_nodes=n_nodes, kind=kind, vocab=np.array(vocab), seeds=np.array([e["seed"] for e in eps]),
             success=np.array([e["success"] for e in eps]), language=np.array([e["text"] for e in eps]), **desc)
    print(f"{a.config} {a.task}: {len(eps)} episodes, {wins} successes, {n_frames} frames, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
