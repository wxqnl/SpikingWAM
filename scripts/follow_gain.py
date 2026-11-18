"""Following gain of the skeleton policy, on the protocol of the Spike-VLA plate-shift counterfactuals
(spike-new-vla-icml/analysis/counterfactual_rollout.py and counterfactual_analysis.py), so that the numbers compare
with theirs (spiking image VLA 0.24, TurboVLA 0.68, at the start of transport, landed within 5 cm).

LIBERO-Spatial, 10 tasks x the first --episodes benchmark initial states. The official episode runs as in
eval_libero (the given inference settings). After it, in the same scene model, branches restore a recorded state, move
only the plate (its free joint, in xy) by (0, 0), (+-1.5 cm, 0), (0, +-1.5 cm) and let the same policy act closed-loop
to success or the episode's remaining step budget:
  t0         the plate moved in the initial state, then the usual 10 settling steps;
  transport  the first policy query once the bowl is grasped and lifted >= 5 cm.
A restore also sets what lies outside the MuJoCo state: the gripper's integrated command and the OSC controller goal,
recorded before every step. Every branch reuses the episode's starting noise (the persistent sample stream), so the
unshifted branch replays the official episode.

Gain: least squares of the bowl's landing shift on the plate's actual shift (both relative to the unshifted branch
from the same state), pooled over episodes, mean of the 2 x 2 diagonal (ideal 1); 95% bootstrap over episodes; also
restricted to branches where the bowl lands within 5 cm of the plate in the branch and in the unshifted branch.

With --perception (11-13), the object nodes come from the perception front end on both cameras, once per plan, as in
eval_libero --perception (default decoding, no smoothing, no hold); the arm nodes stay the simulator's (joint encoders).

  python scripts/follow_gain.py --ckpt runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt --k 1 --flow_steps 1 \
      --fixed_noise --workers 10 --tag k1_steps1_fixed [--perception runs/perception/dr160XL/ckpt.pt]
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

from spikewam.libero_tasks import MAX_STEPS, NUM_STEPS_WAIT, bddl_path, get_suite, make_env, scene_objects

SUITE = "libero_spatial"
DUMMY = [0, 0, 0, 0, 0, 0, -1]
LIFT_TRANSPORT = 0.05
SHIFTS = [(0.0, 0.0), (0.015, 0.0), (-0.015, 0.0), (0.0, 0.015), (0.0, -0.015)]


def snapshot(inner):
    ctrl = inner.robots[0].controller
    cmd = np.asarray(inner.robots[0].gripper.current_action, np.float64)
    return (inner.sim.get_state().flatten().copy(), np.array([-cmd[0], cmd[0]]) if cmd.size == 1 else cmd.copy(),
            (np.array(ctrl.goal_pos, np.float64), np.array(ctrl.goal_ori, np.float64)))


def run_task(task_id, a, queue):
    import torch
    torch.set_num_threads(1)
    from spikewam.policy import SkeletonPolicy, retry_oom
    from spikewam.skeleton import SkeletonExtractor
    policy = retry_oom(SkeletonPolicy, a.ckpt, weights=a.weights, k=a.k, n_exec=a.n_exec, seed=a.seed + task_id,
                       flow_steps=a.flow_steps, fixed_noise=a.fixed_noise)
    suite = get_suite(SUITE)
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    percept = None
    if a.perception:
        from spikewam.perception import FrontEnd
        percept = FrontEnd(a.perception, policy.device)
    env = make_env(bddl_path(task), has_offscreen_renderer=percept is not None)
    inner = env.env
    goal = inner.parsed_problem["goal_state"][0]          # ("on", bowl, plate)
    bowl, plate = goal[1], goal[2]
    budget_all = MAX_STEPS[SUITE]

    def xy(name):
        return inner.sim.data.body_xpos[inner.obj_body_id[name]][:2].copy()

    def act_loop(ext, budget, record=None, cm=None):
        """Closed loop from the current state; record (state, grip, goal, bowl z, grasp) before every step."""
        success, plan, t = False, [], 0
        while t < budget:
            if record is not None:
                record.append((*snapshot(inner), inner.sim.data.body_xpos[inner.obj_body_id[bowl]][2],
                               bool(inner._check_grasp(inner.robots[0].gripper, inner.objects_dict[bowl]))))
            skel = ext(env.sim)
            if percept is not None and not plan:
                skel = percept(env.sim, ext, skel, policy.text_tokens)
                if cm is not None:
                    cm.append(percept.last_cm)
            if not plan:
                plan = list(policy.plan(skel))
            _, _, done, _ = env.step(plan.pop(0).tolist())
            t += 1
            if done:
                success = True
                break
        return success, t

    def branch(ext, state, grip, goal_pg, wait, budget):
        inner.timestep, inner.done = 0, False
        inner.robots[0].gripper.current_action = np.array(grip, dtype=np.float64)
        inner.sim.reset()
        env.set_init_state(state)
        ctrl = inner.robots[0].controller
        ctrl.update(force=True)
        ctrl.goal_pos, ctrl.goal_ori = np.array(goal_pg[0]), np.array(goal_pg[1])
        for _ in range(wait):
            env.step(DUMMY)
        start = np.stack([xy(bowl), xy(plate)])
        success, t = act_loop(ext, budget)
        return {"success": success, "steps": t, "start_xy": start, "end_xy": np.stack([xy(bowl), xy(plate)])}

    qpos = inner.sim.model.get_joint_qpos_addr(inner.objects_dict[plate].joints[0])[0]

    def shifted(state, dxy):
        s = state.copy()
        s[1 + qpos:1 + qpos + 2] += dxy                   # flattened state = [time, qpos, qvel]
        return s

    for ep in range(a.episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        names, ooi = scene_objects(env)
        ext = SkeletonExtractor(env.sim, names, ooi)
        policy.set_scene(ext, task.language)
        noise = policy.noise
        rec = []
        t0_snap = snapshot(inner)
        for _ in range(NUM_STEPS_WAIT):
            env.step(DUMMY)
        z0 = inner.sim.data.body_xpos[inner.obj_body_id[bowl]][2]
        cm = []
        success, steps = act_loop(ext, budget_all, rec, cm)
        out = {"task_id": task_id, "episode": ep, "success": success, "steps": steps,
               "end_xy": np.stack([xy(bowl), xy(plate)]), "branches": []}
        if cm:
            out["percept_cm"] = float(np.mean(cm))
        lifted = [i for i, r in enumerate(rec) if r[4] and r[3] - z0 >= LIFT_TRANSPORT]
        frames = {"t0": None}
        if lifted:      # first policy query at or after the lift (policy steps are counted from the end of settling)
            snap = a.n_exec * int(np.ceil(lifted[0] / a.n_exec))
            if snap < len(rec) - 1:
                frames["transport"] = snap
        out["frames"] = frames
        for name, snap in frames.items():
            for dxy in SHIFTS:
                policy.noise = noise
                if name == "t0":
                    r = branch(ext, shifted(t0_snap[0], dxy), np.zeros(2), t0_snap[2], NUM_STEPS_WAIT, budget_all)
                else:
                    st, grip, gp = rec[snap][:3]
                    r = branch(ext, shifted(st, dxy), grip, gp, 0, budget_all - snap)
                out["branches"].append({"frame": name, "nominal": dxy, **r})
        queue.put(out)
        print(f"task {task_id} ep {ep}: success {success}, frames {frames}, "
              f"branches {[int(b['success']) for b in out['branches']]}", flush=True)
    env.close()


def gains(eps, frame, rng, near=None):
    """counterfactual_analysis.gains of the Spike-VLA analysis (index 0 bowl, 1 plate)."""
    rows = []
    for k, ep in enumerate(eps):
        br = [b for b in ep["branches"] if b["frame"] == frame]
        if not br:
            continue
        ctrl = [b for b in br if tuple(b["nominal"]) == (0.0, 0.0)][0]
        for b in br:
            if b is ctrl:
                continue
            if near is not None and any(np.linalg.norm(x["end_xy"][0] - x["end_xy"][1]) > near for x in (b, ctrl)):
                continue
            rows.append((k, b["start_xy"][1] - ctrl["start_xy"][1], b["end_xy"][0] - ctrl["end_xy"][0]))
    if not rows:
        return None
    ids = np.array([r[0] for r in rows])
    X, Y = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])

    def fit(i):
        coef, *_ = np.linalg.lstsq(X[i], Y[i], rcond=None)
        return np.trace(coef) / 2
    g = fit(np.arange(len(X)))
    uniq = np.unique(ids)
    boots = [fit(np.concatenate([np.flatnonzero(ids == u) for u in rng.choice(uniq, len(uniq))])) for _ in range(1000)]
    return float(g), np.percentile(boots, [2.5, 97.5]).tolist(), len(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", default="ema")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--tasks", default="", help="comma list of task ids (default all 10)")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--flow_steps", type=int, default=0)
    p.add_argument("--fixed_noise", action="store_true")
    p.add_argument("--n_exec", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", default="")
    p.add_argument("--perception", default="", help="checkpoint of spikewam.perception: object nodes from the cameras")
    a = p.parse_args()
    ids = [int(x) for x in a.tasks.split(",")] if a.tasks else list(range(10))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=run_task, args=(t, a, queue)) for t in ids]
    for pr in procs:
        pr.start()
    eps, t0 = [], time.time()
    while len(eps) < len(ids) * a.episodes:
        try:
            eps.append(queue.get(timeout=60))
        except Exception:
            if not any(pr.is_alive() for pr in procs):
                break
    for pr in procs:
        pr.join()
    eps.sort(key=lambda e: (e["task_id"], e["episode"]))
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    out = os.path.join(os.path.dirname(a.ckpt), f"follow_gain_{stem}_{a.weights}{'_' + a.tag if a.tag else ''}")
    with open(out + ".pkl", "wb") as f:
        pickle.dump(eps, f)
    rng = np.random.default_rng(0)
    rep = {"ckpt": a.ckpt, "k": a.k, "flow_steps": a.flow_steps, "fixed_noise": a.fixed_noise,
           "perception": a.perception, "episodes": len(eps),
           "official_success": float(np.mean([e["success"] for e in eps])), "minutes": (time.time() - t0) / 60}
    if any("percept_cm" in e for e in eps):
        rep["percept_cm_official_mean"] = float(np.mean([e["percept_cm"] for e in eps if "percept_cm" in e]))
    for frame in ("t0", "transport"):
        same = []
        for e in eps:
            ctrl = [b for b in e["branches"] if b["frame"] == frame and tuple(b["nominal"]) == (0.0, 0.0)]
            if ctrl:
                same.append((ctrl[0]["success"] == e["success"], float(np.linalg.norm(ctrl[0]["end_xy"][0] -
                                                                                        e["end_xy"][0]))))
        if same:
            rep[f"replay {frame}"] = {"same_success": float(np.mean([s[0] for s in same])),
                                      "landing_diff_max_cm": 100 * max(s[1] for s in same), "n": len(same)}
        for near in (None, 0.05):
            r = gains(eps, frame, rng, near)
            if r is not None:
                rep[f"gain {frame}" + ("" if near is None else " (landed within 5 cm)")] = \
                    {"gain": r[0], "ci": r[1], "n": r[2]}
        br = [b for e in eps for b in e["branches"] if b["frame"] == frame and tuple(b["nominal"]) != (0.0, 0.0)]
        if br:
            rep[f"success under plate shifts ({frame})"] = float(np.mean([b["success"] for b in br]))
    print(json.dumps(rep, indent=1))
    with open(out + ".json", "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", out + ".json")


if __name__ == "__main__":
    main()
