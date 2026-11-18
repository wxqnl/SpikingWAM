"""Does the RoboTwin front end gain from more pixels per object (no training)?

The head view is zoomed by --zoom around the image centre (the camera matrix changed with it, as --geom does in training, so
the zoom is inside the training range for --zoom <= 1 + geom); the wrist views stay as they are. Only objects whose every node
stays inside the zoomed head image are compared. Per object: centre error (cm) and orientation error (deg) of the rigid refit,
original vs zoomed, overall and per kind. If the zoomed input is clearly more accurate, a larger input size would pay.

  CUDA_VISIBLE_DEVICES=1 python scripts/rt_zoom_check.py runs/perception/rtB30tB/ckpt.pt --root data/perception_rtcl8 --zoom 1.2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikewam.perception import PerceptConfig, SpikePercept  # noqa: E402
from spikewam.robotwin.layout import MAX_SLOTS, N_ROBOT as NR, SLOT_SIZE  # noqa: E402
from rt_tta_check import refit  # noqa: E402
from train_perception import IMAGE, DataRT, model_frame_points, prepare  # noqa: E402


def zoom_view(x, M, a, v=0):
    """x (B, V, 3, S, S), M (B, V, 3, 4) on the IMAGE grid: view v zoomed by a around the centre (geom_aug with t = 0)."""
    x, M = x.clone(), M.clone()
    s = x.shape[-1]
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / a
    grid = F.affine_grid(theta, (x.shape[0], 3, s, s), align_corners=False)
    x[:, v] = F.grid_sample(x[:, v], grid, mode="bilinear", padding_mode="border", align_corners=False)
    c = IMAGE / 2
    M[:, v, 0] = a * M[:, v, 0] + (1 - a) * c * M[:, v, 2]
    M[:, v, 1] = a * M[:, v, 1] + (1 - a) * c * M[:, v, 2]
    return x, M


def rot(P, Q):
    """Rotation of the Kabsch fit P -> Q (n, 3) float64."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--root", required=True)
    p.add_argument("--zoom", type=float, default=1.2)
    p.add_argument("--frames", type=int, default=400, help="frames per set at most (evenly spaced)")
    a = p.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    dup_order = getattr(cfg, "dup_order", False)
    data = DataRT(a.root, dev, "spikeminilm16", dup_order, getattr(cfg, "dup_shape", False))
    if getattr(cfg, "text_tokens", False):
        data.load_tokens(dev, "spikeminilm16")
    remap = torch.as_tensor([list(cfg.kinds).index(k) if k in cfg.kinds else 0 for k in data.kinds], device=dev)
    geo_all = model_frame_points(data.skel, data.ids, data.roles, data.mask)
    geo_q = geo_all.clone()
    if dup_order:
        geo_q[data.dup] = 0.0
    rows = []      # (config, kind name, centre err orig, centre err zoom, angle orig, angle zoom)
    for k, name in enumerate(data.names):
        idx = np.flatnonzero(data.task_np == k)
        idx = idx[np.linspace(0, len(idx) - 1, min(a.frames, len(idx))).round().astype(int)]
        cfg_name = name.split("_")[1]
        for kk, images in data.loader(idx, 64, 4):
            i = torch.as_tensor(idx[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0) if getattr(cfg, "text_tokens", False) else (None, None)
            x0, M0 = prepare(images, cfg.size, dev, False), data.M[i]
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i] if dup_order else mask & (gi[..., 3] > 0)
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            est = []
            for z in (1.0, a.zoom):
                x, M = (x0, M0) if z == 1.0 else zoom_view(x0, M0, z)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(x, M, data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                              geo_q[i] if getattr(cfg, "geo", False) else None, tok, tok_mask)
                est.append(refit(o["fused"].float(), gi[..., :3], groups, None).cpu().numpy())
            _, Mz = zoom_view(x0[:, :1], M0[:, :1], a.zoom)
            true = data.skel[i].cpu().numpy()
            h = torch.einsum("bij,bnj->bni", Mz[:, 0, :, :3], data.skel[i]) + Mz[:, 0, None, :, 3]
            px = (h[..., :2] / h[..., 2:3].clamp_min(1e-6)).cpu().numpy()
            inside = ((px > 0) & (px < IMAGE)).all(-1) & (h[..., 2] > 0).cpu().numpy()
            local = gi[..., :3].cpu().numpy()
            kinds = data.kind[i].cpu().numpy()
            for b in range(len(i)):
                for js in groups[b]:
                    js = js.cpu().numpy()
                    if not inside[b, js].all():
                        continue
                    t = true[b, js].astype(np.float64)
                    Rt = rot(local[b, js].astype(np.float64), t)
                    r = [cfg_name, data.kinds[kinds[b, js[0]]]]
                    for e in est:
                        r.append(np.linalg.norm(e[b, js].mean(0) - t.mean(0)) * 100)
                    for e in est:
                        Re = rot(local[b, js].astype(np.float64), e[b, js].astype(np.float64))
                        r.append(np.degrees(np.arccos(np.clip((np.trace(Re @ Rt.T) - 1) / 2, -1, 1))))
                    rows.append(r)
        print(f"{name}: {len(rows)} objects so far", flush=True)
    cfgs = np.array([r[0] for r in rows])
    kind = np.array([r[1] for r in rows])
    v = np.array([r[2:] for r in rows], np.float64)

    def line(label, m):
        if m.sum() < 20:
            return
        c0, c1, a0, a1 = v[m].T
        print(f"  {label:22s} n {m.sum():6d}  centre p50 {np.median(c0):.2f} -> {np.median(c1):.2f}  p90 {np.quantile(c0, .9):.2f} -> {np.quantile(c1, .9):.2f} cm   "
              f"angle p50 {np.median(a0):5.1f} -> {np.median(a1):5.1f}  within 10 deg {np.mean(a0 < 10) * 100:4.0f}% -> {np.mean(a1 < 10) * 100:4.0f}%")

    print(f"\nzoom {a.zoom} on the head view (objects fully inside the zoomed image), original -> zoomed:")
    for c in ("clean", "randomized"):
        line(f"config {c}", cfgs == c)
    for kn in sorted(set(kind)):
        line(kn, kind == kn)


if __name__ == "__main__":
    main()
