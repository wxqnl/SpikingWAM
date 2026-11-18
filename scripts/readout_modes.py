"""Where does the one-step readout lose? Per task, how the policy's own samples spread around its one-step chunk,
against the closed-loop change of the one-step readout (no training).

For every task, --per_task training states (uniform over its steps). Every state gets K chunks of the sampled policy
(10 Euler steps from independent noises) and the one-step chunk (one Euler step from tau = 1: the model's conditional
mean). Over the executed part (steps 0-15, all action dimensions, normalised units):
  spread   mean distance between two sampled chunks;
  between  distance from the one-step chunk to its nearest sample / median distance from a sample to its nearest other
           sample. Samples from one cloud put the mean among them (about 0.7 for Gaussian noise in many dimensions);
           when the samples split into separate ways of doing the task, the mean falls between the groups, far from
           every sample, and the ratio grows.
  between_avg  the same for the average of the K samples (separates split samples from an off-target one-step chunk).
  two_step     between for the chunk of two Euler steps from the same noise.
Each also over the arm dimensions and the gripper dimensions alone ("_arm", "_grip").
Closed loop per task: success of the one-step readout minus that of 10-step MAC k = 3 on the same initial states
(LIBERO: the final 98.35 and 93.90 evaluations, 50 episodes per task; RoboTwin: rt_f2ppt_120k, the 20 paired seeds of
the one-step check against the same seeds of its 100-seed MAC evaluation).

  python scripts/readout_modes.py --bench libero
  python scripts/readout_modes.py --bench robotwin
  python scripts/readout_modes.py --bench robotwin --ckpt runs/rt_f2ppt16_big/ckpt_150000.pt   (offline only)
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from spikewam.data import Normalizer, SkeletonDataset
from spikewam.policy import retry_oom
from spikewam.text import TextEncoder
from spikewam.train import MODELS

N_EXEC = 16
GRIP = {"libero": [6], "robotwin": [6, 13]}
BENCH = {
    "libero": {"ckpt": "runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt",
               "data": os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"),
               "one_step": "runs/f2tm16_geo_finecur16_textmean/eval_ckpt_60000_ema_k1_steps1_fixed.json",
               "mac": "runs/f2tm16_geo_finecur16_textmean/eval_ckpt_60000_ema.json"},
    "robotwin": {"ckpt": "runs/rt_f2ppt_120k/ckpt_120000.pt",
                 "data": "/mnt/data/user01/spikewam_robotwin/demo_clean",
                 "one_step": "runs/robotwin_eval/rt_f2ppt_120k_k1_s1f/demo_clean",
                 "mac": "runs/robotwin_eval/rt_f2ppt_120k/demo_clean"},
}


def integrate(model, args, x, steps, rows=288):
    out = []
    for r in range(0, x.shape[0], rows):
        xr, ar = x[r:r + rows], [a[r:r + rows] for a in args]
        for i in range(steps):
            tau = torch.full((xr.shape[0],), 1 - i / steps, device=x.device)
            xr = xr - retry_oom(model, *ar, xr, tau)[0].float() / steps
        out.append(xr)
    return torch.cat(out)


def closed_loop(bench):
    """{task: one-step success - MAC success} on the same initial states."""
    b = BENCH[bench]
    if bench == "libero":
        one, mac = (json.load(open(b[k]))["tasks"] for k in ("one_step", "mac"))
        return {t: one[t] - mac[t] for t in one}

    def by_seed(path):
        r = json.load(open(path))
        (runs,) = r["results"].values()
        return dict(zip(r["seeds"], [e["success"] for e in runs]))
    out = {}
    for path in sorted(glob.glob(os.path.join(b["one_step"], "*.json"))):
        task = os.path.basename(path)[:-5]
        one, mac = by_seed(path), by_seed(os.path.join(b["mac"], task + ".json"))
        seeds = [s for s in one if s in mac]
        out[task] = float(np.mean([one[s] for s in seeds]) - np.mean([mac[s] for s in seeds]))
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--bench", choices=list(BENCH), required=True)
    pa.add_argument("--per_task", type=int, default=32)
    pa.add_argument("--k", type=int, default=12)
    pa.add_argument("--steps", type=int, default=10)
    pa.add_argument("--ckpt", default="", help="another checkpoint (offline measures only, no closed-loop pairing)")
    pa.add_argument("--out", default="")
    a = pa.parse_args()
    b = dict(BENCH[a.bench], **({"ckpt": a.ckpt} if a.ckpt else {}))
    dev = torch.device("cuda")
    ck = torch.load(b["ckpt"], map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["ema"])
    cfg = model.cfg
    norm = Normalizer(json.load(open(os.path.join(os.path.dirname(b["ckpt"]), "stats.json"))))
    if a.bench == "robotwin":
        ds = SkeletonDataset(b["data"], dev, bench="robotwin", normalizer=norm, posture_rel=cfg.posture_rel,
                             posture_scale=cfg.posture_scale, proprio=cfg.proprio)
        task_of_entry = [n.split("/")[0] for n in ds.task_names]          # one entry per episode
    else:
        ds = SkeletonDataset(b["data"], dev, normalizer=norm)
        task_of_entry = ds.task_names
    text_all, mask_all = TextEncoder(dev, pool=cfg.text_pool, encoder=cfg.text_encoder)(ds.languages)
    entry = ds.task_of.cpu().numpy()
    steps_of = defaultdict(list)
    for t in sorted(set(task_of_entry)):
        steps_of[t] = np.flatnonzero(np.isin(entry, [i for i, n in enumerate(task_of_entry) if n == t]))
    if a.bench == "robotwin":       # states with a previous step (posture-relative chunks and proprio need one)
        ok = set(ds.with_prev.cpu().numpy().tolist())
        steps_of = {t: np.array([i for i in s if i in ok]) for t, s in steps_of.items()}
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=dev).manual_seed(1)
    per_task = {}
    for t, idx in steps_of.items():
        pick = torch.as_tensor(rng.choice(idx, a.per_task, replace=False), device=dev)
        bt = ds.get(pick)
        args = (bt["cur"], bt["node_id"], bt["role"], bt["slot"], bt["node_mask"], text_all[bt["task"]],
                mask_all[bt["task"]])
        n = len(pick)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            noise = torch.randn(n * (a.k + 1), cfg.horizon, cfg.action_dim, device=dev, generator=gen)
            samples = integrate(model, [x.repeat_interleave(a.k, 0) for x in args], noise[:n * a.k], a.steps)
            one = integrate(model, args, noise[n * a.k:], 1)
            two = integrate(model, args, noise[n * a.k:], 2)
        grip = GRIP[a.bench]
        arm = [i for i in range(cfg.action_dim) if i not in grip]
        per_task[t] = {}
        for tag, dims in (("", list(range(cfg.action_dim))), ("_arm", arm), ("_grip", grip)):
            s = samples.view(n, a.k, -1, cfg.action_dim)[:, :, :N_EXEC, dims].flatten(2)   # (n, K, 16 * D)
            m = one[:, None, :N_EXEC, dims].flatten(2)                                     # (n, 1, 16 * D)
            m2 = two[:, None, :N_EXEC, dims].flatten(2)
            d = torch.cdist(s, s)
            off = ~torch.eye(a.k, dtype=torch.bool, device=dev)
            spread = d[:, off].mean(-1)
            nn_ss = d.masked_fill(~off, float("inf")).min(-1).values.median(-1).values.clamp_min(1e-6)
            between = torch.cdist(m, s)[:, 0].min(-1).values / nn_ss
            per_task[t].update({
                "spread" + tag: spread.mean().item(), "between" + tag: between.mean().item(),
                "between_q90" + tag: between.quantile(0.9).item(),
                "between_avg" + tag: (torch.cdist(s.mean(1, keepdim=True), s)[:, 0].min(-1).values / nn_ss).mean().item(),
                "two_step" + tag: (torch.cdist(m2, s)[:, 0].min(-1).values / nn_ss).mean().item(),
                "one_step_vs_mean" + tag: (m[:, 0] - s.mean(1)).norm(dim=-1).mean().item() / spread.mean().item()})
        print(t, json.dumps({k: round(v, 3) for k, v in per_task[t].items()}), flush=True)
    cl = {} if a.ckpt else closed_loop(a.bench)
    tasks = [t for t in per_task if t in cl] if cl else list(per_task)
    from scipy.stats import spearmanr
    res = {"ckpt": b["ckpt"], "per_task_states": a.per_task, "k": a.k, "steps": a.steps,
           "closed_loop_one_step_minus_mac": {t: cl[t] for t in tasks if t in cl},
           "closed_loop_mean_difference": float(np.mean([cl[t] for t in tasks])) if cl else None, "per_task": per_task}
    for key in ("spread", "between", "between_q90", "between_avg") if cl else ():
        rho, p = spearmanr([per_task[t][key] for t in tasks], [cl[t] for t in tasks])
        res[f"spearman_{key}"] = {"rho": float(rho), "p": float(p), "n": len(tasks)}
    for key in [k + tag for k in ("spread", "between", "between_avg", "two_step", "one_step_vs_mean")
                for tag in ("", "_arm", "_grip")]:
        v = np.array([per_task[t][key] for t in tasks])
        res[f"{key}_tasks"] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_task", "closed_loop_one_step_minus_mac")},
                     indent=1))
    tag = "_" + os.path.basename(os.path.dirname(b["ckpt"])) if a.ckpt else ""
    out = a.out or os.path.join(os.environ.get("SW_RUNS", "runs"), f"readout_modes_{a.bench}{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
