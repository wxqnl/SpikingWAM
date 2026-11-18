"""RoboTwin version of the flow-step and consensus checks (consensus_check.py, temporal_check.py; no training).

On fixed RoboTwin training samples (EMA weights, posture-relative chunks and joint-angle input as the checkpoint was
trained with), every state gets K candidate chunks from independent noises, with each number of Euler steps. Errors
against the demonstration over the executed part (steps 0-15): arm joints in radians (RMS), grippers in normalised
units (RMS):
  medoid-k  MAC over k candidates (first h steps, distances in radians, as sample_actions with units);
  mean-k    the average of k candidates; averaged over the K // k disjoint groups.
Posture-relative chunks are compared in radians, so medoid and mean see the same units as in closed loop.

  python scripts/rt_steps_check.py --ckpt runs/rt_f2ppt_120k/ckpt_120000.pt --flow_steps 1 10
"""

import argparse
import json
import os

import torch

from spikewam.data import Normalizer, SkeletonDataset
from spikewam.text import TextEncoder
from spikewam.train import MODELS

ARM = [i for i in range(14) if i not in (6, 13)]
GRIP = [6, 13]
N_EXEC = 16
KS = (1, 3, 12)


def candidates(model, args, k, gen, steps):
    b = args[0].shape[0]
    rep = [x.repeat_interleave(k, 0) for x in args]
    x = torch.randn(b * k, model.cfg.horizon, model.cfg.action_dim, device=args[0].device, generator=gen)
    for i in range(steps):
        tau = torch.full((b * k,), 1 - i / steps, device=x.device)
        x = x - model(*rep, x, tau)[0].float() / steps
    return x.view(b, k, model.cfg.horizon, model.cfg.action_dim)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--flow_steps", type=int, nargs="+", default=[1, 10])
    pa.add_argument("--batches", type=int, default=20)
    pa.add_argument("--batch", type=int, default=32)
    pa.add_argument("--k", type=int, default=12)
    pa.add_argument("--h", type=int, default=10)
    pa.add_argument("--out", default=os.path.join(os.environ.get("SW_RUNS", "runs"), "rt_steps_check.json"))
    a = pa.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["ema"])
    with open(os.path.join(os.path.dirname(a.ckpt), "stats.json")) as f:
        ds = SkeletonDataset(a.data, dev, bench="robotwin", normalizer=Normalizer(json.load(f)),
                             posture_rel=model.cfg.posture_rel, posture_scale=model.cfg.posture_scale,
                             proprio=model.cfg.proprio)
    unit = torch.as_tensor(ds.norm.units(model.cfg.horizon), device=dev)       # radians per normalised unit
    text_all, mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
    gen = torch.Generator(device=dev).manual_seed(0)
    batches = [ds.get(torch.randint(0, ds.n_steps, (a.batch,), device=dev, generator=gen)) for _ in range(a.batches)]
    res = {"ckpt": a.ckpt, "states": a.batches * a.batch}
    for steps in a.flow_steps:
        sq = {f"{m}-{k}": [0.0, 0.0] for m in ("medoid", "mean") for k in KS if not (m == "medoid" and k == 1)}
        g = torch.Generator(device=dev).manual_seed(1)
        for b in batches:
            args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                    mask_all[b["task"]])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = candidates(model, args, a.k, g, steps)
            demo = b["actions"][:, None]
            for k in KS:
                for j in range(a.k // k):
                    xs = x[:, j * k:(j + 1) * k]
                    picks = {"mean": xs.mean(1)}
                    if k > 1:
                        pre = (xs[:, :, :a.h] * unit[:a.h]).flatten(2)
                        best = (torch.cdist(pre, pre) ** 2).sum(-1).argmin(-1)
                        picks["medoid"] = xs[torch.arange(xs.shape[0]), best]
                    for m, c in picks.items():
                        key = f"{m}-{k}"
                        if key not in sq:
                            continue
                        err = (c[:, None] - demo)[:, 0, :N_EXEC]
                        n = a.k // k * len(batches)
                        sq[key][0] += ((err[..., ARM] * unit[:N_EXEC, ARM]) ** 2).mean().item() / n
                        sq[key][1] += (err[..., GRIP] ** 2).mean().item() / n
        res[f"steps={steps}"] = {key: {"arm_deg": float(torch.tensor(v[0]).sqrt().rad2deg()),
                                       "grip": float(torch.tensor(v[1]).sqrt())} for key, v in sq.items()}
        print(steps, json.dumps(res[f"steps={steps}"]), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
