"""Signed per-axis centre error of a RoboTwin front end on the first plans of closed-loop episodes (no training), to see
whether the randomized-scene centre error is a constant offset (e.g. height) rather than noise. Based on rt_slot_errors.py: for every set of a cache,
the plan frames with control step < 16 * --plans, the front end's estimate with the rigid refit, then per object slot the
centre error, the rotation error (Kabsch on the model-frame points, rigid objects only) and the share of frames where the centre
is off by more than 8 cm (another object taken for this one). With --success, episodes are split by the stored success flag.

  CUDA_VISIBLE_DEVICES=7 python scripts/rt_slot_errors.py runs/perception/rtB4g/ckpt.pt --root $SW_DATA/perception_rtcl --config demo_randomized
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
from train_perception import DataRT, model_frame_points, prepare  # noqa: E402


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


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
    rows = []   # config, task, kind, dx, dy, dz (cm), true z (m), table-relative nothing
    for k, name in enumerate(data.names):
        z = np.load(os.path.join(a.root, name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        sel = (z["t"] < 16 * a.plans) & (z["t"] % 16 == 0)
        plan = idx_set[sel]
        est = np.zeros((len(plan), data.skel.shape[1], 3), np.float32)
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                          geo_q[i], tok, tok_mask)
            o = o["fused"].float()
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i]
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            est[kk.numpy()] = refit(o, gi[..., :3], groups, None).cpu().numpy()
        ti = torch.as_tensor(plan, device=dev)
        true, mask = data.skel[ti].cpu().numpy(), data.mask[ti].cpu().numpy()
        slot = ((data.ids[ti] - NR) // SLOT_SIZE).cpu().numpy()
        kind = data.kind[ti].cpu().numpy()
        cfg_name = "clean" if name.startswith("demo_clean") else "randomized"
        for f in range(len(plan)):
            for s in np.unique(slot[f][mask[f]]):
                nodes = np.flatnonzero(mask[f] & (slot[f] == s))
                d = (est[f, nodes].mean(0) - true[f, nodes].mean(0)) * 100
                rows.append((cfg_name, name.split("_", 2)[2], data.kinds[kind[f, nodes[0]]], *d, true[f, nodes].mean(0)[2]))
        print(f"{name}: {len(plan)} frames", flush=True)
    cfgs = np.array([r[0] for r in rows]); d = np.array([r[3:6] for r in rows]); tz = np.array([r[6] for r in rows])
    tasks = np.array([r[1] for r in rows])
    for c in ("clean", "randomized"):
        m = cfgs == c
        print(f"\n{c}: n {m.sum()}  median dx {np.median(d[m, 0]):+.2f} dy {np.median(d[m, 1]):+.2f} dz {np.median(d[m, 2]):+.2f} cm;  "
              f"median |dx| {np.median(abs(d[m, 0])):.2f} |dy| {np.median(abs(d[m, 1])):.2f} |dz| {np.median(abs(d[m, 2])):.2f};  "
              f"true z p10/p50/p90 {np.quantile(tz[m], .1):.3f}/{np.median(tz[m]):.3f}/{np.quantile(tz[m], .9):.3f} m")
        # per task: median signed error per axis
        for t in np.unique(tasks[m]):
            mt = m & (tasks == t)
            md = np.median(d[mt], 0)
            sd = np.median(abs(d[mt] - md), 0)
            print(f"   {t:28s} n {mt.sum():4d} median d ({md[0]:+.2f}, {md[1]:+.2f}, {md[2]:+.2f})  spread ({sd[0]:.2f}, {sd[1]:.2f}, {sd[2]:.2f}) cm")
    if np.any(cfgs == "randomized"):
        m = cfgs == "randomized"
        print(f"\nrandomized: corr(dz, true z) {np.corrcoef(d[m, 2], tz[m])[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
