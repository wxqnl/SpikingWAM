"""Consensus across replanning steps instead of across samples (no training).

The policy plans a 32-step chunk every 16 steps and executes its first 16, so the previous plan's second half covers
the same 16 steps as the current plan's first half. On fixed LIBERO training states i (with i - 16 in the same
episode) both plans are made (one flow step, k = 1, the inference path) and the executed part is blended,
(1 - w) * current[0:16] + w * previous[16:32]: a leaky readout neuron that keeps a share w of the old plan. Errors
are against the demonstration over steps i .. i + 15, six motion dimensions, normalised space (as consensus_check).
  fixed  both plans from the same starting noise (persistent sample stream);
  fresh  independent noises.

  python scripts/temporal_check.py --ckpt runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt
"""

import argparse
import json
import os

import torch

from spikewam.data import SkeletonDataset
from spikewam.text import TextEncoder
from spikewam.train import MODELS

N_EXEC = 16
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5)


def plan(model, b, text_all, mask_all, noise):
    args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]], mask_all[b["task"]])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.sample_actions(*args, k=1, noise=noise).float()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--batches", type=int, default=40)
    pa.add_argument("--batch", type=int, default=64)
    pa.add_argument("--flow_steps", type=int, default=1)
    pa.add_argument("--out", default=os.path.join(os.environ.get("SW_RUNS", "runs"), "temporal_check.json"))
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds = SkeletonDataset(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"), dev)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["ema"])
    model.cfg.flow_steps = a.flow_steps
    md = model.cfg.motion_dims
    text_all, mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
    gen = torch.Generator(device=dev).manual_seed(0)
    ok = torch.nonzero(torch.arange(ds.n_steps, device=dev) - N_EXEC >= ds.ep_start)[:, 0]
    err = {m: {w: [] for w in WEIGHTS} for m in ("fixed", "fresh")}
    for _ in range(a.batches):
        i = ok[torch.randint(0, len(ok), (a.batch,), device=dev, generator=gen)]
        cur, prev = ds.get(i), ds.get(i - N_EXEC)
        shape = (a.batch, model.cfg.horizon, model.cfg.action_dim)
        n0, n1 = (torch.randn(shape, device=dev, generator=gen) for _ in range(2))
        demo = cur["actions"][:, :N_EXEC, :md]
        x_cur = plan(model, cur, text_all, mask_all, n0)[:, :N_EXEC, :md]
        for mode, n_prev in (("fixed", n0), ("fresh", n1)):
            x_prev = plan(model, prev, text_all, mask_all, n_prev)[:, N_EXEC:2 * N_EXEC, :md]
            for w in WEIGHTS:
                err[mode][w].append((((1 - w) * x_cur + w * x_prev - demo) ** 2).mean((1, 2)))
    res = {"ckpt": a.ckpt, "flow_steps": a.flow_steps, "states": a.batches * a.batch,
           **{m: {str(w): torch.cat(v).mean().item() for w, v in e.items()} for m, e in err.items()}}
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
