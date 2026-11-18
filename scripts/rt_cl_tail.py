"""Where the large closed-loop errors of a RoboTwin front end sit (no training).

On a closed-loop cache (spikewam/robotwin/perception_cl.py) the front end's estimate (with the rigid refit, as in evaluation)
is compared with the true skeleton per object slot on the plan frames (control step % 16 == 0). Each object-frame gets:
  hand    the true centre within --near m of either grip site (held or being grasped)
  moved   the true centre more than 3 cm from where it was at the episode's first frame
  lifted  the true centre more than 3 cm above its first-frame height
  fallen  the true centre more than 5 cm below its first-frame height (dropped off the table)
  outcome the episode's stored success flag
and the script reports, per group, its share of the object-frames, the centre error (p50 / p90) and the share of all
errors above 8 cm that falls in the group; then the tasks with the most errors above 8 cm.

  CUDA_VISIBLE_DEVICES=1 python scripts/rt_cl_tail.py runs/perception/rtB12tA/ckpt.pt --root data/perception_rtcl4
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
    p.add_argument("--near", type=float, default=0.06)
    p.add_argument("--max_plans", type=int, default=300, help="plan frames per set at most (evenly spaced)")
    p.add_argument("--big", type=float, default=8.0)
    p.add_argument("--save", default="", help="npz with one row per object-frame")
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
    rows = []      # (set, config, slot, err cm, hand, moved, lifted, fallen, success, plan)
    for k, name in enumerate(data.names):
        z = np.load(os.path.join(a.root, name, "meta.npz"), allow_pickle=True)
        idx_set = np.flatnonzero(data.task_np == k)
        t_all, ep_all = z["t"], z["ep"]
        sel = np.flatnonzero(t_all % 16 == 0)
        if len(sel) > a.max_plans:
            sel = sel[np.linspace(0, len(sel) - 1, a.max_plans).round().astype(int)]
        plan = idx_set[sel]
        first = {e: idx_set[np.flatnonzero(ep_all == e)[np.argmin(t_all[ep_all == e])]] for e in np.unique(ep_all[sel])}
        succ = z["success"] if "success" in z.files else None
        est = np.zeros((len(plan), data.skel.shape[1], 3), np.float32)
        for kk, images in data.loader(plan, 64, 4):
            i = torch.as_tensor(plan[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0) if getattr(cfg, "text_tokens", False) else (None, None)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i],
                          remap[data.kind[i]], geo_q[i] if getattr(cfg, "geo", False) else None, tok, tok_mask)
            o = o["fused"].float()
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i] if dup_order else mask & (gi[..., 3] > 0)
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            est[kk.numpy()] = refit(o, gi[..., :3], groups, None).cpu().numpy()
        ti = torch.as_tensor(plan, device=dev)
        true, mask = data.skel[ti].cpu().numpy(), data.mask[ti].cpu().numpy()
        slot = ((data.ids[ti] - NR) // SLOT_SIZE).cpu().numpy()
        grip = data.grip[ti].cpu().numpy()
        cfg_name = name.split("_")[1]
        for f in range(len(plan)):
            e = ep_all[sel[f]]
            f0 = first[e]
            true0 = data.skel[f0].cpu().numpy()
            for s in np.unique(slot[f][mask[f]]):
                nodes = np.flatnonzero(mask[f] & (slot[f] == s))
                c_true, c_est, c0 = true[f, nodes].mean(0), est[f, nodes].mean(0), true0[nodes].mean(0)
                err = np.linalg.norm(c_est - c_true) * 100
                hand = min(np.linalg.norm(c_true - grip[f, :3]), np.linalg.norm(c_true - grip[f, 4:7])) < a.near
                moved = np.linalg.norm(c_true - c0) > 0.03
                lifted = c_true[2] - c0[2] > 0.03
                fallen = c_true[2] - c0[2] < -0.05
                ok = bool(succ[e]) if succ is not None else True
                rows.append((name, cfg_name, s, err, hand, moved, lifted, fallen, ok, int(t_all[sel[f]]) // 16))
        print(f"{name}: {len(plan)} plan frames", flush=True)
    err = np.array([r[3] for r in rows])
    big = err > a.big
    print(f"\nobject-frames {len(rows)}, centre p50 {np.median(err):.2f} p90 {np.quantile(err, .9):.2f} cm, > {a.big:g} cm {big.mean() * 100:.1f}%")

    def report(label, m):
        m = np.asarray(m, bool)
        if m.sum() == 0:
            return
        print(f"  {label:34s} share {m.mean() * 100:5.1f}%  p50 {np.median(err[m]):5.2f}  p90 {np.quantile(err[m], .9):6.2f}  "
              f">{a.big:g}cm in group {big[m].mean() * 100:5.1f}%  share of all >{a.big:g}cm {(big & m).sum() / max(big.sum(), 1) * 100:5.1f}%")

    col = lambda j: np.array([r[j] for r in rows])
    hand, moved, lifted, fallen, ok, plan_i, cfgs = col(4), col(5), col(6), col(7), col(8), col(9), col(1)
    for c in ("clean", "randomized"):
        report(f"config {c}", cfgs == c)
    report("near a grip site", hand)
    report("not near, moved", ~hand & moved)
    report("not near, not moved", ~hand & ~moved)
    report("lifted", lifted)
    report("fallen (5 cm below start)", fallen)
    report("episode succeeded", ok)
    report("episode failed", ~ok)
    for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 10), (10, 1000)):
        report(f"plan {lo}-{hi - 1}", (plan_i >= lo) & (plan_i < hi))
    report("failed & not near & not moved", ~ok & ~hand & ~moved)
    report("succeeded & not near & not moved", ok & ~hand & ~moved)
    names = np.array([r[0] for r in rows])
    if a.save:
        np.savez(a.save, name=names, cfg=cfgs, slot=col(2), err=err, hand=hand, moved=moved, lifted=lifted, fallen=fallen, ok=ok, plan=plan_i)
    early = plan_i <= 2
    print("\nsets by their large-error rate on the first three plans (before the policy has changed much):")
    us = np.unique(names)
    rate = np.array([big[(names == u) & early].mean() if ((names == u) & early).any() else 0 for u in us])
    for i in np.argsort(-rate)[:20]:
        m = (names == us[i]) & early
        print(f"  {us[i]:45s} early >{a.big:g}cm {rate[i] * 100:5.1f}% (n {m.sum()})  failed episodes {(~ok[names == us[i]]).mean() * 100:4.0f}%")
    print("\nsets with the most errors above the threshold (share of all large errors, rate inside the set):")
    u, cnt = np.unique(names[big], return_counts=True)
    for i in np.argsort(-cnt)[:20]:
        m = names == u[i]
        print(f"  {u[i]:45s} {cnt[i] / big.sum() * 100:5.1f}%  rate {big[m].mean() * 100:5.1f}%  near-hand among its large {hand[m & big].mean() * 100:4.0f}%  "
              f"fallen {fallen[m & big].mean() * 100:4.0f}%  failed-episode {(~ok[m & big]).mean() * 100:4.0f}%")


if __name__ == "__main__":
    main()
