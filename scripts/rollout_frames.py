"""Closed-loop rollouts of the final LIBERO model with the perception front end, keeping what the figure needs at every
replanning step: the camera images the front end read, a 512 x 512 render of the third-person camera for display, the
base-to-pixel camera matrices, the skeleton the policy read (front-end object nodes, joint arm nodes) and the simulator
skeleton. Same loop and settings as the main results (eval_libero / eval_libero_plus with --perception dr4M, PSR: k 1,
1 step, fixed noise, n_exec 16, EMA weights, seed 7); the episode outcome is recorded, nothing else changes.

  LIBERO       python scripts/rollout_frames.py --bench libero --suite libero_spatial --task 0 --episode 0 --out <npz>
  LIBERO-Plus  (source setup/libero_plus_env.sh) python scripts/rollout_frames.py --bench plus --suite libero_spatial --task 2132 --out <npz>
(MUJOCO_GL=osmesa, as the evaluations.)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spikewam.libero_tasks import MAX_STEPS, NUM_STEPS_WAIT, bddl_path, get_suite, make_env, scene_objects  # noqa: E402

DUMMY = [0, 0, 0, 0, 0, 0, -1]
HIRES = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=("libero", "plus"), required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True, help="task id (LIBERO) or task index (LIBERO-Plus)")
    p.add_argument("--episode", type=int, default=0, help="initial state (LIBERO; LIBERO-Plus uses its own first)")
    p.add_argument("--ckpt", default="runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt")
    p.add_argument("--perception", default="runs/perception/dr4M/ckpt.pt")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.bench == "plus":           # as eval_libero_plus: LIBERO-Plus's env (and its MagickWand) loads before the policy's imports
        from libero.libero.envs.env_wrapper import ControlEnv
        from spikewam.eval_libero_plus import bounded_reset
        ControlEnv.reset = bounded_reset
    import torch
    torch.set_num_threads(4)
    from spikewam.perception import FrontEnd
    from spikewam.policy import SkeletonPolicy
    from spikewam.skeleton import SkeletonExtractor
    policy = SkeletonPolicy(a.ckpt, weights="ema", k=1, h=10, n_exec=16, seed=a.seed + a.episode, consensus="medoid",
                            flow_steps=1, fixed_noise=True, temporal=0.0)
    percept = FrontEnd(a.perception, policy.device)
    percept.smooth, percept.hold = 0, False
    percept.model.peak, percept.model.wta = False, False
    stash = {}
    views = percept.views

    def keep_views(sim, ext):
        images, mats = views(sim, ext)
        stash["images"], stash["mats"] = images, mats
        return images, mats
    percept.views = keep_views

    suite = get_suite(a.suite)
    task = suite.get_task(a.task)
    if a.bench == "plus":
        instruction = json.load(open(os.path.join(os.environ["SW_BENCH"], "libero_plus_instructions.json")))["instructions"][task.name]
        env = make_env(bddl_path(task), has_offscreen_renderer=True)
        sensor_noise, env.noise = env.noise, 0
        assert not sensor_noise, "sensor-noise tasks corrupt the front end's image; not handled here"
        env.seed(a.seed)
        env.reset()
        env.set_init_state(suite.get_task_init_states(a.task)[0])
    else:
        instruction = task.language
        env = make_env(bddl_path(task), has_offscreen_renderer=True)
        env.reset()
        env.set_init_state(suite.get_task_init_states(a.task)[a.episode])
    names, ooi = scene_objects(env)
    ext = SkeletonExtractor(env.sim, names, ooi)
    policy.set_scene(ext, instruction)
    front_text = percept.text_of(policy, instruction)
    percept.reset()
    rec = {k: [] for k in ("t", "hires", "images", "mats", "skel", "skel_sim", "cm")}
    success, plan = False, []
    for t in range(MAX_STEPS[a.suite] + NUM_STEPS_WAIT):
        if t < NUM_STEPS_WAIT:
            env.step(DUMMY)
            continue
        skel = ext(env.sim)
        if not plan:
            sim_skel = skel
            skel = percept(env.sim, ext, skel, front_text)
            rec["t"].append(t)
            rec["hires"].append(env.sim.render(camera_name="agentview", width=HIRES, height=HIRES, depth=False)[::-1].copy())
            rec["images"].append(stash["images"])
            rec["mats"].append(stash["mats"])
            rec["skel"].append(skel)
            rec["skel_sim"].append(sim_skel)
            rec["cm"].append(percept.last_cm)
        policy.observe(skel)
        if not plan:
            plan = list(policy.plan(skel))
        _, _, done, _ = env.step(plan.pop(0).tolist())
        if done:
            success = True
            break
    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in rec.items()}, success=success, steps=t + 1, task_name=task.name,
                        instruction=instruction, role=np.asarray(ext.role), slot=np.asarray(ext.slot),
                        node_id=np.asarray(ext.node_id), kinds=np.asarray(ext.kinds), ooi=np.asarray(ooi))
    print(f"{task.name}: success {success}, {t + 1} steps, {len(rec['t'])} plans, front-end error {np.mean(rec['cm']):.2f} cm")


if __name__ == "__main__":
    main()
