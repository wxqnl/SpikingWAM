"""Can the front end learn the randomized table height from RGB (feasibility, a short side training)?

The backbone of a trained front end gets a small head on its pooled feature maps (all three views, 1x1 / 2x2 / 4x4 mean
pooling) that regresses each episode's table offset, labels as in scripts/rt_table_probe.py (from the stored skeleton and the
clean episodes of the same task); backbone and head are trained on the randomized frames of some closed-loop caches and tested
on the first plans of another. Not a model for evaluation: it only says whether the cue is in the images at 160 px.

  CUDA_VISIBLE_DEVICES=1 python scripts/rt_table_head_check.py runs/perception/rtB12tA/ckpt.pt --fit data/perception_rtcl4,data/perception_rtcl6 --test data/perception_rtcl5
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikewam.perception import PerceptConfig, SpikePercept  # noqa: E402
from rt_table_probe import offsets  # noqa: E402
from train_perception import DataRT, prepare  # noqa: E402


def frames_with_offset(data, root, plans=None):
    off = offsets(data, root)
    idx, y = [], []
    for k, name in enumerate(data.names):
        if not name.startswith("demo_randomized"):
            continue
        z = np.load(os.path.join(root, name, "meta.npz"), allow_pickle=True)
        fr = np.flatnonzero(data.task_np == k)
        sel = np.ones(len(fr), bool) if plans is None else (z["t"] < 16 * plans) & (z["t"] % 16 == 0)
        for f, e in zip(fr[sel], z["ep"][sel]):
            if (k, int(e)) in off:
                idx.append(f); y.append(off[(k, int(e))])
    return np.array(idx), np.array(y, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--fit", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev)
    model.load_state_dict(st["model"])
    bb = model.backbone.train()
    fits = []
    for root in a.fit.split(","):
        d = DataRT(root, dev, "spikeminilm16")
        idx, y = frames_with_offset(d, root)
        fits.append((d, idx, torch.as_tensor(y, device=dev)))
        print(f"fit {root}: {len(idx)} randomized frames, offset mean {y.mean() * 100:+.2f} sd {y.std() * 100:.2f} cm", flush=True)
    dt = DataRT(a.test, dev, "spikeminilm16")
    tidx, ty = frames_with_offset(dt, a.test, plans=3)
    print(f"test {a.test}: {len(tidx)} first-plan frames; no information |error| p50 {np.median(abs(ty - np.concatenate([f[2].cpu().numpy() for f in fits]).mean())) * 100:.2f} cm", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        c = bb(torch.zeros(3, 3, cfg.size, cfg.size, device=dev)).shape[1]
    head = nn.Sequential(nn.Linear(3 * 21 * c, 256), nn.GELU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW([{"params": bb.parameters(), "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    iters = [iter(d.loader(idx, a.batch, 6, steps=a.steps)) for d, idx, _ in fits]

    def predict(x):
        feat = bb(x.flatten(0, 1)).float()
        pooled = torch.cat([F.adaptive_avg_pool2d(feat, g).flatten(1) for g in (1, 2, 4)], 1)
        return head(pooled.view(x.shape[0], -1))[:, 0] / 100          # metres

    def test():
        bb.eval(); pr = []
        for kk, images in dt.loader(tidx, 64, 4):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pr.append(predict(prepare(images, cfg.size, dev, False)).float().cpu().numpy())
        bb.train()
        e = np.concatenate(pr) - ty
        return np.median(abs(e)) * 100, abs(e).mean() * 100

    for step in range(1, a.steps + 1):
        d, idx, y = fits[step % len(fits)]
        kk, images = next(iters[step % len(fits)])
        target = y[torch.as_tensor(kk, device=dev)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = predict(prepare(images, cfg.size, dev, True))
        loss = (pred.float() - target).abs().mean() * 100
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"step {step} train |error| {loss.item():.2f} cm", flush=True)
        if step % 1000 == 0:
            p50, mean = test()
            print(f"  test |error| p50 {p50:.2f} mean {mean:.2f} cm", flush=True)


if __name__ == "__main__":
    main()
