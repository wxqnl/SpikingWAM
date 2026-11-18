"""Does averaging two RoboTwin front ends help, and which objects still go wrong (no training)?

On closed-loop caches that none of the given front ends trained on, every frame of a subset is estimated by each front end
(rigid refit as in evaluation) and by their ensemble (the fused node estimates averaged, then the same refit). Per object:
centre error (cm) and orientation error (deg); per frame: the mean object-node error (as percept_cm) and its share above 2 cm.
Reported per config, and per kind for the first front end and the ensemble.

  CUDA_VISIBLE_DEVICES=2 python scripts/rt_ensemble_check.py runs/perception/rtB30tB/ckpt.pt,runs/perception/rtB21tB/ckpt.pt --root data/perception_rtcl9
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
from spikewam.robotwin.layout import MAX_SLOTS, N_ROBOT as NR, SLOT_SIZE  # noqa: E402
from rt_tta_check import refit  # noqa: E402
from rt_zoom_check import rot  # noqa: E402
from train_perception import DataRT, model_frame_points, prepare  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", help="comma list of front-end checkpoints (same object kinds)")
    p.add_argument("--root", required=True)
    p.add_argument("--frames", type=int, default=150, help="frames per set at most (evenly spaced)")
    a = p.parse_args()
    dev = "cuda"
    models, cfgs = [], []
    for c in a.ckpts.split(","):
        st = torch.load(c, map_location="cpu", weights_only=False)
        cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
        m = SpikePercept(cfg).to(dev).eval()
        m.load_state_dict(st["model"])
        models.append(m), cfgs.append(cfg)
    cfg = cfgs[0]
    assert all(tuple(c.kinds) == tuple(cfg.kinds) for c in cfgs), "object kinds differ"
    dup_order = getattr(cfg, "dup_order", False)
    data = DataRT(a.root, dev, "spikeminilm16", dup_order, getattr(cfg, "dup_shape", False))
    data.load_tokens(dev, "spikeminilm16")
    remap = torch.as_tensor([list(cfg.kinds).index(k) if k in cfg.kinds else 0 for k in data.kinds], device=dev)
    geo_all = model_frame_points(data.skel, data.ids, data.roles, data.mask)
    geo_q = geo_all.clone()
    if dup_order:
        geo_q[data.dup] = 0.0
    names = [f"m{i}" for i in range(len(models))] + ["ens"]
    obj_rows = []      # (config, kind, [centre per name], [angle per name])
    frame_rows = []    # (config, [frame mean error per name])
    for k, name in enumerate(data.names):
        idx = np.flatnonzero(data.task_np == k)
        idx = idx[np.linspace(0, len(idx) - 1, min(a.frames, len(idx))).round().astype(int)]
        cfg_name = name.split("_")[1]
        for kk, images in data.loader(idx, 64, 4):
            i = torch.as_tensor(idx[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0)
            x = prepare(images, cfg.size, dev, False)
            fused = []
            for m, c in zip(models, cfgs):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(x, data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                          geo_q[i] if getattr(c, "geo", False) else None, tok if getattr(c, "text_tokens", False) else None,
                          tok_mask if getattr(c, "text_tokens", False) else None)
                fused.append(o["fused"].float())
            fused.append(torch.stack(fused).mean(0))
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i] if dup_order else mask & (gi[..., 3] > 0)
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            est = [refit(f, gi[..., :3], groups, None).cpu().numpy() for f in fused]
            true, local, m_np = data.skel[i].cpu().numpy(), gi[..., :3].cpu().numpy(), mask.cpu().numpy()
            kinds = data.kind[i].cpu().numpy()
            for b in range(len(i)):
                mb = m_np[b]
                frame_rows.append((cfg_name, [np.linalg.norm(e[b, mb] - true[b, mb], axis=-1).mean() * 100 for e in est]))
                for js in groups[b]:
                    js = js.cpu().numpy()
                    t = true[b, js].astype(np.float64)
                    Rt = rot(local[b, js].astype(np.float64), t)
                    cen = [np.linalg.norm(e[b, js].mean(0) - t.mean(0)) * 100 for e in est]
                    ang = []
                    for e in est:
                        Re = rot(local[b, js].astype(np.float64), e[b, js].astype(np.float64))
                        ang.append(np.degrees(np.arccos(np.clip((np.trace(Re @ Rt.T) - 1) / 2, -1, 1))))
                    obj_rows.append((cfg_name, data.kinds[kinds[b, js[0]]], cen, ang))
        print(f"{name}: {len(frame_rows)} frames, {len(obj_rows)} objects so far", flush=True)
    print(f"\nfront ends: {a.ckpts}  (ens = mean of the fused estimates, then refit)")
    fc = np.array([r[0] for r in frame_rows]); fv = np.array([r[1] for r in frame_rows])
    for c in ("clean", "randomized"):
        v = fv[fc == c]
        if len(v) == 0:
            continue
        print(f"  frames {c:10s} n {len(v):6d}  " + "  ".join(f"{n}: mean {v[:, q].mean():.2f} p50 {np.median(v[:, q]):.2f} >2cm {np.mean(v[:, q] > 2) * 100:4.1f}%"
                                                         for q, n in enumerate(names)))
    oc = np.array([r[0] for r in obj_rows]); ok = np.array([r[1] for r in obj_rows])
    cen = np.array([r[2] for r in obj_rows]); ang = np.array([r[3] for r in obj_rows])
    e = len(names) - 1
    for c in ("clean", "randomized"):
        s = oc == c
        if s.sum() == 0:
            continue
        print(f"\n  objects {c}: kind, n, centre p50 / p90 (m0 -> ens) cm, angle p50 / p90 (m0 -> ens) deg, sorted by m0 angle p90")
        rows = []
        for kn in sorted(set(ok[s])):
            m = s & (ok == kn)
            if m.sum() < 30:
                continue
            rows.append((np.quantile(ang[m, 0], .9), kn, m.sum(), np.median(cen[m, 0]), np.median(cen[m, e]), np.quantile(cen[m, 0], .9), np.quantile(cen[m, e], .9),
                         np.median(ang[m, 0]), np.median(ang[m, e]), np.quantile(ang[m, 0], .9), np.quantile(ang[m, e], .9)))
        for r in sorted(rows, reverse=True)[:18]:
            print("    %-16s %5d  centre %.2f -> %.2f / %.2f -> %.2f   angle %5.1f -> %5.1f / %5.1f -> %5.1f" % r[1:])


if __name__ == "__main__":
    main()
