"""Would a per-episode temporal filter on the RoboTwin front end's object estimates help (no training)?

On closed-loop caches (the front end in the loop) every plan frame (control step % 16 == 0) of an episode is estimated as in
evaluation (rigid refit), in time order. The filter only uses what the front end has at run time (its own estimates and the grip
sites from the joints):
  near    the object's estimated centre within --near m of a grip site now or at the previous plan: no filtering, history reset
  far     the estimate within --jump m of the filtered position: added to the history (last --k accepted), output = history mean
          farther: an outlier, output = the filtered position; after --hold outliers in a row the new place is accepted (moved)
Reported per config: mean object-node error per frame (as percept_cm) raw vs filtered, its p50 / p90 and the share above 3 cm.

  CUDA_VISIBLE_DEVICES=1 python scripts/rt_temporal_check.py runs/perception/rtB30tB/ckpt.pt --root <dir of finished rtcl8 sets>
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


def filtered(est, slot, mask, grip, a):
    """est (T, n, 3) one episode's plan-frame estimates in time order; slot (n,), mask (n,) object nodes; grip (T, 8).
    Returns the filtered estimates (T, n, 3)."""
    out = est.copy()
    for s in np.unique(slot[mask]):
        j = np.flatnonzero(mask & (slot == s))
        hist, pos, out_n, near_prev = [], None, 0, True
        for f in range(len(est)):
            cur = est[f, j]
            c = cur.mean(0)
            near = min(np.linalg.norm(c - grip[f, :3]), np.linalg.norm(c - grip[f, 4:7])) < a.near
            if near or near_prev or pos is None:
                hist, pos, out_n = [cur], cur, 0
            elif np.linalg.norm(c - pos.mean(0)) < a.jump:
                hist = (hist + [cur])[-a.k:]
                pos, out_n = np.mean(hist, 0), 0
            else:
                out_n += 1
                if out_n >= a.hold:
                    hist, pos, out_n = [cur], cur, 0
            out[f, j] = pos
            near_prev = near
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--root", required=True)
    p.add_argument("--near", type=float, default=0.10)
    p.add_argument("--jump", type=float, default=0.03)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--hold", type=int, default=2)
    p.add_argument("--max_eps", type=int, default=30, help="episodes per set at most")
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
    res = {}      # config -> (raw errors, filtered errors) per frame, cm
    per_set = []
    for k, name in enumerate(data.names):
        z = np.load(os.path.join(a.root, name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        t_all, ep_all = z["t"], z["ep"]
        eps = np.unique(ep_all)[:a.max_eps]
        sel = np.flatnonzero((t_all % 16 == 0) & np.isin(ep_all, eps))
        sel = sel[np.lexsort((t_all[sel], ep_all[sel]))]
        plan = idx_set[sel]
        est = np.zeros((len(plan), data.skel.shape[1], 3), np.float32)
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0) if getattr(cfg, "text_tokens", False) else (None, None)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i],
                          remap[data.kind[i]], geo_q[i] if getattr(cfg, "geo", False) else None, tok, tok_mask)
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i] if dup_order else mask & (gi[..., 3] > 0)
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            est[kk.numpy()] = refit(o["fused"].float(), gi[..., :3], groups, None).cpu().numpy()
        ti = torch.as_tensor(plan, device=dev)
        true, mask = data.skel[ti].cpu().numpy(), data.mask[ti].cpu().numpy()
        slot = ((data.ids[ti] - NR) // SLOT_SIZE).cpu().numpy()
        grip = data.grip[ti].cpu().numpy()
        cfg_name = name.split("_")[1]
        r0, r1 = [], []
        for e in eps:
            f = np.flatnonzero(ep_all[sel] == e)
            if len(f) == 0:
                continue
            m = mask[f[0]]
            fe = filtered(est[f], slot[f[0]], m, grip[f], a)
            for q in range(len(f)):
                r0.append(np.linalg.norm(est[f[q], m] - true[f[q], m], axis=-1).mean() * 100)
                r1.append(np.linalg.norm(fe[q, m] - true[f[q], m], axis=-1).mean() * 100)
        res.setdefault(cfg_name, ([], []))
        res[cfg_name][0].extend(r0)
        res[cfg_name][1].extend(r1)
        per_set.append((name, np.mean(r0), np.mean(r1), len(r0)))
        print(f"{name}: {len(r0)} plan frames, mean {np.mean(r0):.2f} -> {np.mean(r1):.2f} cm", flush=True)
    print(f"\nfilter: near {a.near} m, jump {a.jump} m, history {a.k}, hold {a.hold}; per-frame mean object-node error, raw -> filtered")
    for c, (r0, r1) in res.items():
        r0, r1 = np.array(r0), np.array(r1)
        print(f"  {c:10s} frames {len(r0):6d}  mean {r0.mean():.2f} -> {r1.mean():.2f}  p50 {np.median(r0):.2f} -> {np.median(r1):.2f}  "
              f"p90 {np.quantile(r0, .9):.2f} -> {np.quantile(r1, .9):.2f}  >3cm {np.mean(r0 > 3) * 100:.1f}% -> {np.mean(r1 > 3) * 100:.1f}%")


if __name__ == "__main__":
    main()
