"""Diagnose whether RoboTwin's spiking front end loses objects while held or after release/fall (no training).
Use every stored frame, rigid refit as in rt_axis_bias.py, and TRUE skeleton centres/grips to label object state.
Tables count object-frames; episode counts are unique episodes across slots. JSON also records each episode/slot's
first held error > 5 cm, whether that object fell later, and its last-held error. Timing is observational, not causal proof.

  CUDA_VISIBLE_DEVICES=2 python scripts/rt_hold_check.py runs/perception/rtB30tCM/ckpt.pt --root data/perception_rtcl10 --tasks scan_object,handover_block,hanging_mug,stack_blocks_three --out runs/perception/checks/hold_rtB30tCM_rtcl10.json
"""

from __future__ import annotations

import argparse
import json
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


def analyse(name, true, est, mask, slot, grip, ep, t, success):
    """Centres average all valid nodes of a slot, exactly as in rt_axis_bias.py. Distances are metres until e."""
    ct = np.full((len(ep), MAX_SLOTS, 3), np.nan)
    ce = np.full_like(ct, np.nan)
    for f in range(len(ep)):
        for s in np.unique(slot[f][mask[f]]):
            nodes = mask[f] & (slot[f] == s)
            ct[f, s], ce[f, s] = true[f, nodes].mean(0), est[f, nodes].mean(0)
    error = np.linalg.norm(ce - ct, axis=-1) * 100
    present = np.isfinite(ct).all(-1)
    assert np.isfinite(error[present]).all(), f"{name}: nonfinite estimated centre"
    state = np.full(error.shape, "", dtype="<U5")
    records = []
    # DataRT builds grip as [g0 xyz, width0, g1 xyz, width1] from the TRUE robot skeleton.
    assert grip.shape == (len(ep), 8), f"{name}: unexpected gripper layout"
    grips = np.stack([grip[:, :3], grip[:, 4:7]], axis=1)
    for e in np.unique(ep):
        frames = np.flatnonzero(ep == e)
        frames = frames[np.argsort(t[frames], kind="stable")]
        for s in range(MAX_SLOTS):
            j = frames[present[frames, s]]
            if not len(j):
                continue
            assert j[0] == frames[0], f"{name}: episode {e}, slot {s} absent at first frame"
            c, c0 = ct[j, s], ct[frames[0], s]
            rest = np.linalg.norm(c - c0, axis=-1) < 0.01
            near = np.linalg.norm(c[:, None] - grips[j], axis=-1).min(1) < 0.08
            held = (c[:, 2] > c0[2] + 0.015) & near & ~rest
            fallen = c[:, 2] < c0[2] - 0.05
            state[j, s] = np.where(rest, "rest", np.where(held, "held", "moved"))
            lost = j[held & (error[j, s] > 5)]
            last = j[held][-1] if held.any() else None
            falls = j[fallen]
            records.append(dict(
                set=name, ep=int(e), slot=s, success=bool(success[e]),
                lost_while_held=bool(len(lost)),
                first_lost_t=int(t[lost[0]]) if len(lost) else None,
                first_lost_error_cm=float(error[lost[0], s]) if len(lost) else None,
                fallen=bool(len(falls)), first_fallen_t=int(t[falls[0]]) if len(falls) else None,
                fallen_after_loss=bool(len(lost) and np.any(t[falls] > t[lost[0]])),
                last_held_t=int(t[last]) if last is not None else None,
                last_held_error_cm=float(error[last, s]) if last is not None else None,
            ))
    return dict(name=name, config=name.split("_")[1], task=name.split("_", 2)[2],
                error=error, state=state, success=success[ep], records=records)


def summarize(sets, success):
    errors = np.concatenate([d["error"][d["success"] == success].ravel() for d in sets]) if sets else np.empty(0)
    states = np.concatenate([d["state"][d["success"] == success].ravel() for d in sets]) if sets else np.empty(0, dtype=str)
    records = [r for d in sets for r in d["records"] if r["success"] == success]
    count = lambda rs: len({(r["set"], r["ep"]) for r in rs})
    stats = {}
    for s in ("rest", "held", "moved"):
        v = errors[states == s]
        stats[s] = dict(frames=len(v), median_cm=float(np.median(v)) if len(v) else None,
                        p90_cm=float(np.quantile(v, 0.9)) if len(v) else None)
    held = errors[states == "held"]
    return dict(success=success, episodes=count(records), object_episodes=len(records), states=stats,
                held_gt3_share=float(np.mean(held > 3)) if len(held) else None,
                held_gt5_share=float(np.mean(held > 5)) if len(held) else None,
                lost_while_held_episodes=count(r for r in records if r["lost_while_held"]),
                fallen_episodes=count(r for r in records if r["fallen"]),
                lost_then_fallen_episodes=count(r for r in records if r["fallen_after_loss"]))


def print_table(title, rows, per_set=False):
    print(f"\n{title} (object-frames; errors in cm; empty groups: -)")
    columns = ["config"] + (["task"] if per_set else [])
    columns += ["success", "episodes", "rest n / median / p90", "held n / median / p90", "moved n / median / p90",
                "held >3 cm", "held >5 cm", "lost held eps", "fallen eps", "lost then fell eps"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        cells = [r["config"]] + ([r["task"]] if per_set else [])
        cells += [str(r["success"]), str(r["episodes"])]
        for s in ("rest", "held", "moved"):
            v = r["states"][s]
            cells.append(f'{v["frames"]} / {v["median_cm"]:.2f} / {v["p90_cm"]:.2f}' if v["frames"] else "0 / - / -")
        cells += [f"{r[k] * 100:.1f}%" if r[k] is not None else "-" for k in ("held_gt3_share", "held_gt5_share")]
        cells += [str(r[k]) for k in ("lost_while_held_episodes", "fallen_episodes", "lost_then_fallen_episodes")]
        print("| " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt")
    p.add_argument("--root", required=True, help="one RoboTwin closed-loop cache root")
    p.add_argument("--tasks", help="optional comma-separated task names")
    p.add_argument("--max_sets", type=int, default=0, help="sets after task filtering (0: all)")
    p.add_argument("--out", required=True, help="JSON output path")
    a = p.parse_args()
    if a.max_sets < 0 or "," in a.root:
        p.error("--max_sets must be nonnegative and --root must be one cache root")
    tasks = {t.strip() for t in a.tasks.split(",") if t.strip()} if a.tasks is not None else None
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
    selected = [(k, name) for k, name in enumerate(data.names) if tasks is None or name.split("_", 2)[2] in tasks]
    selected = selected[:a.max_sets or None]
    if not selected:
        p.error("no sets match --tasks")
    sets = []
    for k, name in selected:
        idx_set = np.flatnonzero(data.task_np == k)
        with np.load(os.path.join(a.root, name, "meta.npz"), allow_pickle=True) as z:
            ep, t, success = z["ep"], z["t"], z["success"].astype(bool)
        assert len(ep) == len(t) == len(idx_set), f"{name}: meta/frame length mismatch"
        assert ep.ndim == t.ndim == success.ndim == 1
        assert len(ep) and ep.min() >= 0 and ep.max() < len(success), f"{name}: invalid episode indices"
        est = np.zeros((len(idx_set), data.skel.shape[1], 3), np.float32)
        for kk, images in data.loader(idx_set, 64, 4):
            i = torch.as_tensor(idx_set[kk.numpy()], device=dev)
            tok, tok_mask = data.tokens_of(data.task[i], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i],
                          data.grip[i], remap[data.kind[i]], geo_q[i], tok, tok_mask)
            mask, gi = data.mask[i], geo_all[i]
            slot = ((data.ids[i] - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1)
            fit = mask & (gi[..., 3] > 0) & ~data.dup[i]
            groups = [[torch.nonzero(fit[b] & (slot[b] == s))[:, 0] for s in range(MAX_SLOTS)] for b in range(len(i))]
            groups = [[js for js in g if len(js) >= 3] for g in groups]
            est[kk.numpy()] = refit(o["fused"].float(), gi[..., :3], groups, None).cpu().numpy()
        ti = torch.as_tensor(idx_set, device=dev)
        true, mask = data.skel[ti].cpu().numpy(), data.mask[ti].cpu().numpy()
        slot = ((data.ids[ti] - NR) // SLOT_SIZE).cpu().numpy()
        sets.append(analyse(name, true, est, mask, slot, data.grip[ti].cpu().numpy(), ep, t, success))
        print(f"{name}: {len(idx_set)} stored frames, {len(np.unique(ep))} episodes", flush=True)
    per_set = [dict(config=d["config"], task=d["task"], **summarize([d], ok)) for d in sets for ok in (True, False)]
    summary = [dict(config=c, **summarize([d for d in sets if d["config"] == c], ok))
               for c in ("clean", "randomized") for ok in (True, False)]
    print_table("Per set", per_set, per_set=True)
    print_table("All selected sets", summary)
    result = dict(ckpt=a.ckpt, root=a.root, sets=[d["name"] for d in sets],
                  frame_unit="object-frame", episode_unit="unique (set, ep), any slot",
                  thresholds_cm=dict(rest_displacement_lt=1, held_rise_gt=1.5, held_grip_distance_lt=8,
                                     fallen_drop_gt=5, lost_held_error_gt=5),
                  per_set=per_set, summary=summary, episode_slots=[r for d in sets for r in d["records"]])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"\nSaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
