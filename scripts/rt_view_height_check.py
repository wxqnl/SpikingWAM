"""Per-view height and position error of a RoboTwin front end on the first plans of closed-loop episodes (no training):
for each camera (head, left wrist, right wrist) the node height error of that view's own z, the error of that view's cast 3D
point, and its share of the fused confidence; clean against randomized (the randomized table is lowered by 0-3 cm).
Also the error of the per-episode table offset if it were known: true table offset from the lowest resting object nodes.

  CUDA_VISIBLE_DEVICES=0 python scripts/rt_view_height_check.py runs/perception/rtB30t/ckpt.pt --root data/perception_rtcl4
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
from train_perception import DataRT, model_frame_points, prepare  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--root", required=True)
    p.add_argument("--plans", type=int, default=3)
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
    rec = []                                    # per node: config, task, episode, node id, true z, fused dz, head dz
    res = {"clean": [], "randomized": []}       # per node: dz per view (3), |dxyz| per view (3), weight per view (3), fused dz, |fused|, episode mean true z offset
    for k, name in enumerate(data.names):
        z = np.load(os.path.join(a.root, name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        sel = (z["t"] < 16 * a.plans) & (z["t"] % 16 == 0)
        plan = idx_set[sel]
        c = "clean" if name.startswith("demo_clean") else "randomized"
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                          geo_q[i], tok, tok_mask)
            xyz = torch.stack(o["xyz"], 0).float()                  # (V, B, N, 3)
            w = torch.stack(o["conf"], 0).float().softmax(0)
            true = data.skel[i]
            m = data.mask[i]
            dz = (xyz[..., 2] - true[None, ..., 2]) * 100
            de = (xyz - true[None]).norm(dim=-1) * 100
            fz = (o["fused"].float()[..., 2] - true[..., 2]) * 100
            fe = (o["fused"].float() - true).norm(dim=-1) * 100
            rows = torch.cat([dz.permute(1, 2, 0), de.permute(1, 2, 0), w.permute(1, 2, 0), fz[..., None], fe[..., None]], -1)[m]
            res[c].append(rows.cpu().numpy())
            ep = torch.as_tensor(z["ep"][sel][kk.numpy()], device=dev)[:, None].expand_as(m)
            for row in zip(ep[m].tolist(), data.ids[i][m].tolist(), true[..., 2][m].tolist(), fz[m].tolist(), dz[0][m].tolist()):
                rec.append((c, name.split("_", 2)[2], *row))
        print(f"{name}: {len(plan)} frames", flush=True)
    names = ["head", "left wrist", "right wrist"]
    for c, rs in res.items():
        r = np.concatenate(rs)
        print(f"\n{c}: {len(r)} object nodes")
        for v in range(3):
            print(f"  {names[v]:12s} own z: median {np.median(r[:, v]):+.2f} |dz| p50 {np.median(abs(r[:, v])):.2f} p90 {np.quantile(abs(r[:, v]), .9):.2f}   "
                  f"cast point p50 {np.median(r[:, 3 + v]):.2f} p90 {np.quantile(r[:, 3 + v], .9):.2f} cm   fused weight mean {r[:, 6 + v].mean():.2f}")
        print(f"  fused        |dz| p50 {np.median(abs(r[:, 9])):.2f} p90 {np.quantile(abs(r[:, 9]), .9):.2f}   point p50 {np.median(r[:, 10]):.2f} p90 {np.quantile(r[:, 10], .9):.2f} cm")
        best = np.min(r[:, 3:6], 1)
        print(f"  best single view per node (oracle choice): point p50 {np.median(best):.2f} p90 {np.quantile(best, .9):.2f} cm")
    # table offset per randomized episode: median over its nodes of (true z - the clean median true z of the same task and node id)
    import collections
    ref = collections.defaultdict(list)
    for c, t, e, nid, tz, fz, hz in rec:
        if c == "clean":
            ref[(t, nid)].append(tz)
    ref = {k: np.median(v) for k, v in ref.items()}
    per_ep = collections.defaultdict(list)
    for c, t, e, nid, tz, fz, hz in rec:
        if c == "randomized" and (t, nid) in ref:
            per_ep[(t, e)].append(tz - ref[(t, nid)])
    off = {k: np.median(v) for k, v in per_ep.items()}
    x, yf, yh = [], [], []
    for c, t, e, nid, tz, fz, hz in rec:
        if c == "randomized" and (t, e) in off:
            x.append(off[(t, e)] * 100); yf.append(fz); yh.append(hz)
    x, yf, yh = np.array(x), np.array(yf), np.array(yh)
    o = np.array(list(off.values())) * 100
    print(f"\nrandomized table offset from the clean reference: {len(o)} episodes, p10/p50/p90 {np.quantile(o, .1):+.2f}/{np.median(o):+.2f}/{np.quantile(o, .9):+.2f} cm")
    for lab, y in (("fused", yf), ("head", yh)):
        A = np.stack([x, np.ones_like(x)], 1)
        k, b0 = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - (k * x + b0)
        print(f"  {lab:6s} dz = {k:+.2f} * offset {b0:+.2f};  |dz| p50 {np.median(abs(y)):.2f} -> {np.median(abs(resid)):.2f} cm after removing the offset term")


if __name__ == "__main__":
    main()
