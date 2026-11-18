"""Offline check on held-out RoboTwin frames, no training: does a per-object rigid refit and/or test-time jitter (n random zooms and
shifts of every view with matching camera matrices, per-node median) lower the front end's error?

  CUDA_VISIBLE_DEVICES=7 python scripts/rt_tta_check.py runs/perception/rt160h/ckpt.pt --root $D/perception_rt,$D/perception_rt2 --n 4,8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikewam.perception import PerceptConfig, SpikePercept  # noqa: E402
from train_perception import DataRT, geom_aug, model_frame_points, prepare  # noqa: E402


def refit(est, local, ok, w0=None):
    """est (B, n, 3); local (B, n, 3) model-frame points; ok (B, n) nodes with a model point; slots from the caller. Kabsch + Huber 2 cm.
    w0 (B, n): optional prior weight of each node (its confidence)."""
    out = est.clone()
    for b in range(est.shape[0]):
        for js in ok[b]:
            P, Q = local[b, js].double(), est[b, js].double()
            prior = torch.ones(len(js), dtype=torch.float64, device=est.device) if w0 is None else w0[b, js].double().clamp_min(1e-3)
            w = prior.clone()
            for _ in range(4):
                pc, qc = (w[:, None] * P).sum(0) / w.sum(), (w[:, None] * Q).sum(0) / w.sum()
                U, _, Vt = torch.linalg.svd(((P - pc) * w[:, None]).T @ (Q - qc))
                d = torch.sign(torch.det(Vt.T @ U.T))
                D = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64, device=est.device))
                R = Vt.T @ D @ U.T
                Fp = (P - pc) @ R.T + qc
                w = prior / torch.clamp((Fp - Q).norm(dim=1) / 0.02, min=1.0)
            out[b, js] = Fp.float()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--root", required=True)
    p.add_argument("--n", default="4,8")
    p.add_argument("--strength", type=float, default=0.1)
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--workspace", type=float, default=0.0, help="leave out object nodes farther than this (m, horizontal) from the table centre or below the floor")
    p.add_argument("--all", action="store_true", help="every frame of the sets by config (sets the model never saw), not the held-out episodes")
    a = p.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    data = DataRT(a.root, dev, "spikeminilm16")
    from spikewam.robotwin.layout import MAX_SLOTS, N_ROBOT as NR, SLOT_SIZE
    names = list(cfg.kinds)
    remap = torch.as_tensor([names.index(k) if k in names else 0 for k in data.kinds], device=dev)
    geo_all = model_frame_points(data.skel, data.ids, data.roles, data.mask)
    ns = sorted({0} | {int(x) for x in a.n.split(",") if int(x) > 0})
    torch.manual_seed(0)
    groups_all = zip(data.roots, data.val_by_root)
    if a.all:
        cfgs = np.array([n.split("_")[1] for n in data.names])[data.task_np]
        groups_all = [(c, np.flatnonzero(cfgs == c)) for c in ("clean", "randomized")]
    for split, idx in groups_all:
        if len(idx) == 0:
            continue
        idx = idx[np.linspace(0, len(idx) - 1, min(a.frames, len(idx))).astype(int)]
        errs = {(n, r): [] for n in ns for r in (0, 1)}
        for k, images in data.loader(idx, 64, 4):
            i = torch.as_tensor(idx[k.numpy()], device=dev)
            x0 = prepare(images, a.size, dev, False)
            M0, text = data.M[i], data.text[data.task[i], 0]
            kind = remap[data.kind[i]]
            geo = geo_all[i] if getattr(cfg, "geo", False) else None
            skel, mask = data.skel[i], data.mask[i]
            if a.workspace:
                mask = mask & (skel[..., :2].norm(dim=-1) < a.workspace) & (skel[..., 2] > -0.1)
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            gi = geo_all[i]
            groups = [[torch.nonzero(mask[b] & (slot[b] == s) & (gi[b, :, 3] > 0))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            outs = []
            for n in range(max(ns) + 1):
                x, M = (x0, M0) if n == 0 else geom_aug(x0, M0, a.strength)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outs.append(model(x, M, text, data.ids[i], data.roles[i], data.grip[i], kind, geo)["fused"].float())
            for n in ns:
                est = outs[0] if n == 0 else torch.stack(outs[1:n + 1]).median(0).values
                for r in (0, 1):
                    e = refit(est, gi[..., :3], groups) if r else est
                    errs[(n, r)].append((e - skel).norm(dim=-1)[mask].cpu())
        print(split, flush=True)
        for (n, r), v in errs.items():
            e = torch.cat(v) * 100
            print(f"  jitter {n} refit {r}: mean {e.mean():.2f} p50 {e.median():.2f} p90 {e.quantile(0.9):.2f} >5cm {(e > 5).float().mean() * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
