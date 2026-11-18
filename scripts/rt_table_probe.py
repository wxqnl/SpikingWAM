"""Is the randomized table height readable from the front end's head-camera features (no training of the network)?

RoboTwin's randomized config lowers the table by 0-3 cm and the front end's height error follows the offset almost one to one
(scripts/rt_view_height_check.py). Here a ridge probe on the frozen backbone's head-view feature map (mean pooled over 1x1,
2x2 and 4x4 grids) is fitted to each episode's table offset on one closed-loop cache and tested on another. The offset of an
episode is the median over its object nodes of (true height at the first plan - the clean episodes' median height of the
same task and node), so no simulator state beyond the stored skeleton is used.

  CUDA_VISIBLE_DEVICES=0 python scripts/rt_table_probe.py runs/perception/rtB30t/ckpt.pt --fit data/perception_rtcl4 --test data/perception_rtcl5
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikewam.perception import PerceptConfig, SpikePercept  # noqa: E402
from train_perception import DataRT, prepare  # noqa: E402


def offsets(data, root):
    """{(set index, episode): table offset (m)} for the randomized sets, from the clean sets of the same task."""
    ref = collections.defaultdict(list)
    first = {}
    for k, name in enumerate(data.names):
        z = np.load(os.path.join(root, name, "meta.npz"), allow_pickle=True)
        idx = np.flatnonzero(data.task_np == k)
        t0 = z["t"] == 0
        for f in idx[t0]:
            m = data.mask[f].cpu().numpy()
            ids, zz = data.ids[f].cpu().numpy()[m], data.skel[f, :, 2].cpu().numpy()[m]
            task = name.split("_", 2)[2]
            if name.startswith("demo_clean"):
                for i, h in zip(ids, zz):
                    ref[(task, i)].append(h)
            else:
                first[(k, int(z["ep"][f - idx[0]]))] = (task, ids, zz)
    ref = {key: np.median(v) for key, v in ref.items()}
    out = {}
    for key, (task, ids, zz) in first.items():
        d = [h - ref[(task, i)] for i, h in zip(ids, zz) if (task, i) in ref]
        if d:
            out[key] = float(np.median(d))
    return out


def features(model, cfg, data, root, plans, dev):
    off = offsets(data, root)
    X, y = [], []
    for k, name in enumerate(data.names):
        if not name.startswith("demo_randomized"):
            continue
        z = np.load(os.path.join(root, name, "meta.npz"), allow_pickle=True)
        idx = np.flatnonzero(data.task_np == k)
        sel = (z["t"] < 16 * plans) & (z["t"] % 16 == 0)
        frames, eps = idx[sel], z["ep"][sel]
        keep = np.array([(k, int(e)) in off for e in eps])
        frames, eps = frames[keep], eps[keep]
        for kk, images in data.loader(frames, 64, 4):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = prepare(images, cfg.size, dev, False)
                feat = model.backbone(x.flatten(0, 1))[::cfg.views].float()          # head view (B, C, h, w)
            pooled = [F.adaptive_avg_pool2d(feat, g).flatten(1) for g in (1, 2, 4)]
            X.append(torch.cat(pooled, 1).cpu().numpy())
            y.extend(off[(k, int(e))] for e in eps[kk.numpy()])
        print(f"{name}: {len(frames)} frames", flush=True)
    return np.concatenate(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--fit", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--plans", type=int, default=3)
    a = p.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    sets = {}
    for tag, root in (("fit", a.fit), ("test", a.test)):
        data = DataRT(root, dev, "spikeminilm16", getattr(cfg, "dup_order", False), getattr(cfg, "dup_shape", False))
        sets[tag] = features(model, cfg, data, root, a.plans, dev)
    (Xa, ya), (Xb, yb) = sets["fit"], sets["test"]
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
    ym = ya.mean()
    K = Xa @ Xa.T
    print(f"\nfit {len(ya)} frames, test {len(yb)} frames, {Xa.shape[1]} features; offset fit mean {ym * 100:+.2f} sd {ya.std() * 100:.2f} cm")
    print(f"no information (predict the fit mean): test |error| p50 {np.median(abs(yb - ym)) * 100:.2f} mean {abs(yb - ym).mean() * 100:.2f} cm")
    for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
        alpha = np.linalg.solve(K + lam * np.eye(len(K)), ya - ym)
        pred_a = K @ alpha + ym
        pred_b = Xb @ (Xa.T @ alpha) + ym
        r2 = 1 - ((yb - pred_b) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
        print(f"  ridge {lam:8.0e}: fit |error| p50 {np.median(abs(ya - pred_a)) * 100:.2f}  test |error| p50 {np.median(abs(yb - pred_b)) * 100:.2f} mean {abs(yb - pred_b).mean() * 100:.2f} cm  R2 {r2:+.2f}")


if __name__ == "__main__":
    main()
