"""Can the wrist cameras' side view fix the randomized table height at inference (no training)?

At the start of an episode the head camera looks down on the table (good x, y; the height of a resting object depends on the
table, which the network barely infers), while the two wrist cameras look across the table (a side view, where a pixel row is
close to a height). For every object node whose head-based estimate projects inside a wrist image (margin --margin px of 256)
and in front of it, the head ray (head pixel) and the wrist ray (wrist pixel) are triangulated (closest point between the two
rays); the episode's height correction is the median over such nodes and the first --plans plans of (triangulated height -
fused height), and every object node is moved along the head ray by it. Reports the nodes' height error and centre error
before and after, by config, and the correction against the needed one (true minus fused height, episode median).

  CUDA_VISIBLE_DEVICES=0 python scripts/rt_wrist_height_check.py runs/perception/rtB12tA/ckpt.pt --root data/perception_rtcl4
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikewam.perception import BINS_UV, IMAGE, PerceptConfig, SpikePercept  # noqa: E402
from train_perception import DataRT, model_frame_points, prepare  # noqa: E402


def centre(M):
    return -np.linalg.solve(M[:, :3], M[:, 3])


def ray(M, uv):
    """Unit direction of pixel uv's ray in the frame of M (3, 4)."""
    d = np.linalg.solve(M[:, :3], np.array([uv[0], uv[1], 1.0]))
    return d / np.linalg.norm(d)


def closest(c1, d1, c2, d2):
    """Midpoint of the shortest segment between two rays."""
    w = c1 - c2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    if s <= 0 or t <= 0:
        return None
    return (c1 + s * d1 + c2 + t * d2) / 2


def project(M, p):
    h = M[:, :3] @ p + M[:, 3]
    return h[:2] / h[2], h[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--root", required=True)
    p.add_argument("--plans", type=int, default=3)
    p.add_argument("--margin", type=float, default=16)
    a = p.parse_args()
    dev = "cuda"
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
    out = collections.defaultdict(list)      # config -> per node (dz before, dz after, |centre| before, after)
    corr = []                                # randomized: (estimated correction, needed correction, nodes used)
    for k, name in enumerate(data.names):
        z = np.load(os.path.join(a.root, name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        sel = (z["t"] < 16 * a.plans) & (z["t"] % 16 == 0)
        plan, eps = idx_set[sel], z["ep"][sel]
        c = "clean" if name.startswith("demo_clean") else "randomized"
        fused, uvs = np.zeros((len(plan), data.skel.shape[1], 3)), np.zeros((3, len(plan), data.skel.shape[1], 2))
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                          geo_q[i], tok, tok_mask)
            fused[kk.numpy()] = o["fused"].float().cpu().numpy()
            for v in range(3):
                uvs[v, kk.numpy()] = torch.stack([(o["logit_u"][v].float().softmax(-1) * uvb).sum(-1), (o["logit_v"][v].float().softmax(-1) * uvb).sum(-1)], -1).cpu().numpy()
        Ms = data.M[torch.as_tensor(plan, device=dev)].cpu().numpy().astype(np.float64)
        true, mask = data.skel[torch.as_tensor(plan, device=dev)].cpu().numpy(), data.mask[torch.as_tensor(plan, device=dev)].cpu().numpy()
        for e in np.unique(eps):
            fr = np.flatnonzero(eps == e)
            dzs = []
            for f in fr:
                ch = centre(Ms[f, 0])
                for j in np.flatnonzero(mask[f]):
                    for v in (1, 2):
                        px, depth = project(Ms[f, v], fused[f, j])
                        if depth <= 0 or not (a.margin < px[0] < IMAGE - a.margin and a.margin < px[1] < IMAGE - a.margin):
                            continue
                        q = closest(ch, ray(Ms[f, 0], uvs[0, f, j]), centre(Ms[f, v]), ray(Ms[f, v], uvs[v, f, j]))
                        if q is not None and abs(q[2] - fused[f, j, 2]) < 0.06:
                            dzs.append(q[2] - fused[f, j, 2])
            need = np.median(np.concatenate([true[f, mask[f], 2] - fused[f, mask[f], 2] for f in fr]))
            est = float(np.median(dzs)) if len(dzs) >= 3 else 0.0
            if c == "randomized":
                corr.append((est, need, len(dzs)))
            for f in fr:
                ch = centre(Ms[f, 0])
                p0 = fused[f, mask[f]]
                t = (p0[:, 2] + est - ch[2]) / (p0[:, 2] - ch[2])
                p1 = ch + t[:, None] * (p0 - ch)
                tt = true[f, mask[f]]
                for b0, b1, tr in zip(p0, p1, tt):
                    out[c].append(((b0[2] - tr[2]) * 100, (b1[2] - tr[2]) * 100, np.linalg.norm(b0 - tr) * 100, np.linalg.norm(b1 - tr) * 100))
        print(f"{name}: {len(plan)} frames", flush=True)
    for c, r in out.items():
        r = np.array(r)
        print(f"\n{c}: {len(r)} nodes  |dz| p50 {np.median(abs(r[:, 0])):.2f} -> {np.median(abs(r[:, 1])):.2f} cm, p90 {np.quantile(abs(r[:, 0]), .9):.2f} -> {np.quantile(abs(r[:, 1]), .9):.2f};  "
              f"point p50 {np.median(r[:, 2]):.2f} -> {np.median(r[:, 3]):.2f}, p90 {np.quantile(r[:, 2], .9):.2f} -> {np.quantile(r[:, 3], .9):.2f} cm")
    cr = np.array(corr)
    if len(cr):
        print(f"\nrandomized episodes {len(cr)}: wrist nodes used p50 {np.median(cr[:, 2]):.0f}, none in {(cr[:, 2] < 3).mean() * 100:.0f}%;  "
              f"correction - needed |p50| {np.median(abs(cr[:, 0] - cr[:, 1])) * 100:.2f} cm (needed |p50| {np.median(abs(cr[:, 1])) * 100:.2f}), corr {np.corrcoef(cr[:, 0], cr[:, 1])[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
