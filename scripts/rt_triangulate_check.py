"""Does triangulating the per-view pixels fix the randomized-scene height error (no training)?

The front end casts each view's pixel onto the plane of its predicted height z, so a wrong height (RoboTwin's randomized
config lowers the table by up to 3 cm, which the network barely infers) also moves the point along the camera ray. Here the
per-view pixels (u, v) of the same model are triangulated instead: weighted least squares over the views, each view's two
equations scaled by its fused-softmax confidence and 1 / depth (so residuals are in pixels), views whose pixel lies outside the
image left out, plus an optional height prior lam * (z - z_pred) (lam in pixels per metre). Compared on the first plans of
closed-loop episodes after the rigid refit (as in evaluation): centre error, per-axis error, rotation of rigid objects.

  CUDA_VISIBLE_DEVICES=0 python scripts/rt_triangulate_check.py runs/perception/rtB30t/ckpt.pt --root data/perception_rtcl4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikewam.perception import BINS_UV, IMAGE, PerceptConfig, SpikePercept  # noqa: E402
from spikewam.robotwin.layout import MAX_SLOTS, N_ROBOT as NR, SLOT_SIZE  # noqa: E402
from rt_slot_errors import kabsch  # noqa: E402
from rt_tta_check import refit  # noqa: E402
from train_perception import DataRT, model_frame_points, prepare  # noqa: E402


def triangulate(M, u, v, z, w, lam):
    """M (B, V, 3, 4); u, v, z, w (V, B, N); returns (B, N, 3) and the number of views used (B, N)."""
    Mv = M.permute(1, 0, 2, 3)[:, :, None]                          # (V, B, 1, 3, 4)
    ok = (u > 0) & (u < IMAGE) & (v > 0) & (v < IMAGE)
    rows, rhs, wts = [], [], []
    for coord, r in ((u, 0), (v, 1)):
        a = Mv[..., r, :3] - coord[..., None] * Mv[..., 2, :3]      # (V, B, N, 3)
        b = -(Mv[..., r, 3] - coord * Mv[..., 2, 3])
        rows.append(a); rhs.append(b); wts.append(w * ok)
    A = torch.cat(rows, 0).permute(1, 2, 0, 3)                      # (B, N, 2V, 3)
    bb = torch.cat(rhs, 0).permute(1, 2, 0)
    W = torch.cat(wts, 0).permute(1, 2, 0)
    # residuals in pixels: divide each row by the point's depth in that view (from a first unweighted-depth solve)
    for it in range(2):
        if it == 1:
            Mz = M.permute(1, 0, 2, 3)[:, :, 2]                     # (V, B, 4)
            depth = torch.einsum("vbk,bnk->vbn", Mz[..., :3], X) + Mz[..., 3:4]   # (V, B, N)
            d = torch.cat([depth, depth], 0).permute(1, 2, 0).clamp_min(0.05)
            Wd = W / d ** 2
        else:
            Wd = W
        AtA = (A * Wd[..., None]).transpose(-1, -2) @ A
        Atb = ((A * Wd[..., None]).transpose(-1, -2) @ bb[..., None])[..., 0]
        if lam > 0:
            e = torch.tensor([0.0, 0.0, 1.0], device=A.device)
            zp = (z * w).sum(0) / w.sum(0).clamp_min(1e-9)          # (B, N) confidence-weighted predicted height
            scale = lam ** 2
            AtA = AtA + scale * (e[:, None] * e[None, :])
            Atb = Atb + scale * zp[..., None] * e
        AtA = AtA + 1e-9 * torch.eye(3, device=A.device)
        X = torch.linalg.solve(AtA, Atb)
    return X, ok.sum(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--root", required=True)
    p.add_argument("--plans", type=int, default=3)
    p.add_argument("--lams", default="0,30,100,300")
    p.add_argument("--tasks", default="", help="comma list (default: all)")
    a = p.parse_args()
    dev = "cuda"
    lams = [float(x) for x in a.lams.split(",")]
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    data = DataRT(a.root, dev, "spikeminilm16", getattr(cfg, "dup_order", False), getattr(cfg, "dup_shape", False))
    data.load_tokens(dev, "spikeminilm16")
    remap = torch.as_tensor([list(cfg.kinds).index(k) if k in cfg.kinds else 0 for k in data.kinds], device=dev)
    geo_all = model_frame_points(data.skel, data.ids, data.roles, data.mask)
    geo_q = geo_all.clone()
    geo_q[data.dup] = 0.0
    uvb = (torch.arange(BINS_UV, device=dev).float() + 0.5) / 2
    zb = cfg.z_lo + (torch.arange(cfg.bins_z, device=dev).float() + 0.5) * (cfg.z_hi - cfg.z_lo) / cfg.bins_z
    want = set(a.tasks.split(",")) if a.tasks else None
    methods = ["fused"] + [f"tri{l:g}" for l in lams]
    rows = {m: [] for m in methods}     # (config, task, kind, centre err, dx, dy, dz, rot)
    nviews = []
    for k, name in enumerate(data.names):
        task = name.split("_", 2)[2]
        if want and task not in want:
            continue
        z = np.load(os.path.join(a.root, name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        sel = (z["t"] < 16 * a.plans) & (z["t"] % 16 == 0)
        plan = idx_set[sel]
        est = {m: np.zeros((len(plan), data.skel.shape[1], 3), np.float32) for m in methods}
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                          geo_q[i], tok, tok_mask)
            with torch.no_grad():
                u = torch.stack([(x.float().softmax(-1) * uvb).sum(-1) for x in o["logit_u"]], 0)
                v = torch.stack([(x.float().softmax(-1) * uvb).sum(-1) for x in o["logit_v"]], 0)
                zz = torch.stack([(x.float().softmax(-1) * zb).sum(-1) for x in o["logit_z"]], 0)
                w = torch.stack(o["conf"], 0).float().softmax(0)
                outs = {"fused": o["fused"].float()}
                for l in lams:
                    X, nv = triangulate(data.M[i].float(), u, v, zz, w, l)
                    outs[f"tri{l:g}"] = torch.where((nv >= 2)[..., None] | (l > 0), X, o["fused"].float())
                nviews.append(nv[data.mask[i]].cpu().numpy())
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i]
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            for m in methods:
                est[m][kk.numpy()] = refit(outs[m], gi[..., :3], groups, None).cpu().numpy()
        ti = torch.as_tensor(plan, device=dev)
        true, mask = data.skel[ti].cpu().numpy(), data.mask[ti].cpu().numpy()
        slot = ((data.ids[ti] - NR) // SLOT_SIZE).cpu().numpy()
        kind, geo = data.kind[ti].cpu().numpy(), geo_all[ti].cpu().numpy()
        cfg_name = "clean" if name.startswith("demo_clean") else "randomized"
        for f in range(len(plan)):
            for s in np.unique(slot[f][mask[f]]):
                nodes = np.flatnonzero(mask[f] & (slot[f] == s))
                g = nodes[geo[f, nodes, 3] > 0]
                for m in methods:
                    d = (est[m][f, nodes].mean(0) - true[f, nodes].mean(0)) * 100
                    rot = np.nan
                    if len(g) >= 3:
                        R = kabsch(true[f, g], est[m][f, g])
                        rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                    rows[m].append((cfg_name, task, data.kinds[kind[f, nodes[0]]], np.linalg.norm(d), *d, rot))
        print(f"{name}: {len(plan)} frames", flush=True)
    nv = np.concatenate(nviews)
    print(f"\nobject nodes seen inside the image by: " + ", ".join(f"{c} views {(nv == c).mean() * 100:.1f}%" for c in range(4)))
    for c in ("clean", "randomized"):
        print(f"\n{c}")
        for m in methods:
            r = [x for x in rows[m] if x[0] == c]
            if not r:
                continue
            e = np.array([x[3] for x in r]); d = np.array([x[4:7] for x in r]); rot = np.array([x[7] for x in r])
            print(f"  {m:8s} n {len(e)}  centre p50 {np.median(e):.2f} p90 {np.quantile(e, .9):.2f} >2cm {(e > 2).mean() * 100:4.1f}% >8cm {(e > 8).mean() * 100:4.1f}%  "
                  f"|dx| {np.median(abs(d[:, 0])):.2f} |dy| {np.median(abs(d[:, 1])):.2f} |dz| {np.median(abs(d[:, 2])):.2f}  rot p50 {np.nanmedian(rot):.1f}")
        tasks = sorted({x[1] for x in rows["fused"] if x[0] == c})
        print(f"  per task, centre p50 cm  ({' / '.join(methods)})")
        for t in tasks:
            vals = []
            for m in methods:
                e = np.array([x[3] for x in rows[m] if x[0] == c and x[1] == t])
                vals.append(f"{np.median(e):5.2f}")
            print(f"    {t:28s} " + " / ".join(vals))


if __name__ == "__main__":
    main()
