"""Per-object error of a RoboTwin front end on the first plans of closed-loop episodes (no training): for every set of a cache,
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
    p.add_argument("--config", default="demo_randomized")
    p.add_argument("--plans", type=int, default=1, help="first this many plans of each episode")
    p.add_argument("--tasks", default="", help="comma list (default: all)")
    p.add_argument("--success", action="store_true")
    p.add_argument("--conf_weight", action="store_true", help="refit with each node weighted by its best view's confidence")
    p.add_argument("--by_plan", action="store_true", help="split by plan index instead of success")
    a = p.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    data = DataRT(a.root, dev, "spikeminilm16", getattr(cfg, "dup_order", False))
    if getattr(cfg, "text_tokens", False):
        data.load_tokens(dev, "spikeminilm16")
    remap = torch.as_tensor([list(cfg.kinds).index(k) if k in cfg.kinds else 0 for k in data.kinds], device=dev)
    geo_all = model_frame_points(data.skel, data.ids, data.roles, data.mask)
    geo_q = geo_all.clone()
    if getattr(cfg, "dup_order", False):
        geo_q[data.dup] = 0.0
    want = set(a.tasks.split(",")) if a.tasks else None
    total = {}
    for k, name in enumerate(data.names):
        if not name.startswith(a.config) or (want and name[len(a.config) + 1:] not in want):
            continue
        z = np.load(os.path.join(a.root.split(",")[0], name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        sel = z["t"] < 16 * a.plans
        sel &= z["t"] % 16 == 0
        plan, ep = idx_set[sel], z["ep"][sel]
        succ = z["success"] if ("success" in z.files and a.success) else None
        ests = []
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                          geo_q[i] if getattr(cfg, "geo", False) else None, tok, tok_mask)
            w0 = torch.stack(o["conf"], 0).float().sigmoid().amax(0) if a.conf_weight else None
            o = o["fused"].float()
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i] if getattr(cfg, "dup_order", False) else mask & (gi[..., 3] > 0)
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            ests.append((kk.numpy(), refit(o, gi[..., :3], groups, w0).cpu().numpy()))
        est = np.zeros((len(plan), data.skel.shape[1], 3), np.float32)
        for kk, e in ests:
            est[kk] = e
        ti = torch.as_tensor(plan, device=dev)
        true, mask = data.skel[ti].cpu().numpy(), data.mask[ti].cpu().numpy()
        slot = ((data.ids[ti] - NR) // SLOT_SIZE).cpu().numpy()
        kind, geo = data.kind[ti].cpu().numpy(), geo_all[ti].cpu().numpy()
        rows = {}
        for f in range(len(plan)):
            group = "all" if succ is None else ("succ" if succ[ep[f]] else "fail")
            if a.by_plan:
                group = f"p{int(z['t'][sel][f]) // 16}"
            for s in np.unique(slot[f][mask[f]]):
                nodes = np.flatnonzero(mask[f] & (slot[f] == s))
                c_err = np.linalg.norm(est[f, nodes].mean(0) - true[f, nodes].mean(0)) * 100
                n_err = np.linalg.norm(est[f, nodes] - true[f, nodes], axis=-1).mean() * 100
                g = nodes[geo[f, nodes, 3] > 0]
                rot = np.nan
                if len(g) >= 3:
                    R = kabsch(true[f, g], est[f, g])
                    rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                key = (s, data.kinds[kind[f, nodes[0]]], group)
                rows.setdefault(key, []).append((c_err, n_err, rot))
        print(f"== {name[len(a.config) + 1:]}  ({len(np.unique(ep))} episodes, {len(plan)} frames)")
        for key in sorted(rows):
            v = np.array(rows[key])
            total.setdefault(key[2], []).append(v)
            print(f"   slot {key[0]} {key[1][:28]:28s} {key[2]:4s} n {len(v):3d} centre p50 {np.median(v[:, 0]):5.2f} p90 {np.quantile(v[:, 0], .9):6.2f} "
                  f"node mean {np.median(v[:, 1]):5.2f}  rot p50 {np.nanmedian(v[:, 2]) if np.isfinite(v[:, 2]).any() else float('nan'):5.1f} deg  >8cm {(v[:, 0] > 8).mean() * 100:4.0f}%"
                  + (f"  rot <15 {(v[:, 2] < 15).mean() * 100:3.0f}% 15-150 {((v[:, 2] >= 15) & (v[:, 2] <= 150)).mean() * 100:3.0f}% >150 {(v[:, 2] > 150).mean() * 100:3.0f}%" if np.isfinite(v[:, 2]).any() else ""), flush=True)
    for g, vs in total.items():
        v = np.concatenate(vs)
        print(f"ALL {g}: n {len(v)} centre p50 {np.median(v[:, 0]):.2f} p90 {np.quantile(v[:, 0], .9):.2f} >8cm {(v[:, 0] > 8).mean() * 100:.1f}% rot p50 {np.nanmedian(v[:, 2]):.1f}")


if __name__ == "__main__":
    main()
