"""Offline check of averaging a still object's estimate over plans (no training): on the policy's own closed loop (perception_rtcl),
only the plan frames (control step a multiple of 16), the front end's estimate per plan, rigid refit per object, then per
object a running mean of the refit nodes that restarts when the new estimate's centre leaves the mean by more than --gate.
Error of every object node inside the workspace, split into objects that are still (moved < 1 cm since the previous plan)
and moving.

  CUDA_VISIBLE_DEVICES=7 python scripts/rt_track_check.py runs/perception/rtB4g_cl/ckpt.pt --root $SW_DATA/perception_rtcl
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--root", required=True)
    p.add_argument("--gates", default="0.01,0.02,0.04")
    p.add_argument("--max_n", type=int, default=0, help="average at most this many plans (0: all since the restart)")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--sets", type=int, default=0, help="only the first this many sets (0: all)")
    a = p.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    data = DataRT(a.root, dev, "spikeminilm16")
    remap = torch.as_tensor([list(cfg.kinds).index(k) if k in cfg.kinds else 0 for k in data.kinds], device=dev)
    geo_all = model_frame_points(data.skel, data.ids, data.roles, data.mask)
    gates = [float(g) for g in a.gates.split(",")]
    errs = {("raw", s): [] for s in ("still", "moving")}
    errs.update({(g, s): [] for g in gates for s in ("still", "moving")})
    for k, name in enumerate(data.names[:a.sets or None]):
        z = np.load(os.path.join(a.root.split(",")[0], name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        plan = idx_set[z["t"] % a.stride == 0]
        ep = z["ep"][z["t"] % a.stride == 0]
        est = torch.zeros(len(plan), data.skel.shape[1], 3, device=dev)
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]],
                          geo_all[i] if getattr(cfg, "geo", False) else None)["fused"].float()
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            groups = [[torch.nonzero(mask[b] & (slot[b] == s) & (gi[b, :, 3] > 0))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            est[torch.as_tensor(kk.numpy(), device=dev)] = refit(o, gi[..., :3], groups)
        est, true = est.cpu().numpy(), data.skel[torch.as_tensor(plan, device=dev)].cpu().numpy()
        mask = data.mask[torch.as_tensor(plan, device=dev)].cpu().numpy()
        slot = ((data.ids[torch.as_tensor(plan, device=dev)] - NR) // SLOT_SIZE).cpu().numpy()
        inside = (np.linalg.norm(true[..., :2], axis=-1) < 0.8) & (true[..., 2] > -0.1) & mask
        for e in np.unique(ep):
            j = np.flatnonzero(ep == e)
            moved = np.zeros((len(j), true.shape[1]), bool)
            moved[1:] = np.linalg.norm(true[j[1:]] - true[j[:-1]], axis=-1) > 0.01
            state = {g: {} for g in gates}
            for n, f in enumerate(j):
                cat = np.where(moved[n], "moving", "still")
                for s in ("still", "moving"):
                    sel = inside[f] & (cat == s)
                    errs[("raw", s)].append(np.linalg.norm(est[f] - true[f], axis=-1)[sel])
                for g in gates:
                    out = est[f].copy()
                    for s_id in np.unique(slot[f][mask[f]]):
                        nodes = np.flatnonzero(mask[f] & (slot[f] == s_id))
                        cur = est[f][nodes]
                        mean, cnt = state[g].get(s_id, (None, 0))
                        if mean is None or np.linalg.norm(cur.mean(0) - mean.mean(0)) > g:
                            mean, cnt = cur, 1
                        else:
                            cnt = cnt + 1 if not a.max_n else min(cnt + 1, a.max_n)
                            mean = mean + (cur - mean) / cnt
                        state[g][s_id] = (mean, cnt)
                        out[nodes] = mean
                    for s in ("still", "moving"):
                        sel = inside[f] & (cat == s)
                        errs[(g, s)].append(np.linalg.norm(out - true[f], axis=-1)[sel])
    for key, v in errs.items():
        e = np.concatenate(v) * 100
        print(f"{str(key[0]):>5} {key[1]:>6}: n {len(e)} mean {e.mean():.2f} p50 {np.median(e):.2f} p90 {np.quantile(e, 0.9):.2f} <1cm {(e < 1).mean() * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
