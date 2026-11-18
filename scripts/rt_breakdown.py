"""Where does the RoboTwin action error sit? (implementation record 4.10; LIBERO versions: loss_breakdown.py and
copy_error.py)

On the same fixed RoboTwin training samples for every checkpoint (EMA weights):
  chunk  error of the sampled chunk (flow_steps Euler steps from noise, k = 1) over the executed part (steps 0-15),
         in radians for the arm joints (RMS) and in normalised units for the grippers;
  toggle the gripper on the samples whose demonstrated gripper opens or closes within the executed part (it moves
         more than TOGGLE normalised units between step 0 and step 15): the step at which the sampled gripper crosses
         the midpoint against the step at which the demonstration does (late = positive), and how often it does not
         cross by step 15. A model that copies the gripper it reads (proprio) would cross late or not at all;
  moves  the step-15 arm error split by whether the demonstrated joint moves more than BIG_MOVE degrees from q_t by
         step 15 (rare large moves, e.g. wrist rolls) or not, per joint, and on the large moves the sampled move
         against the demonstrated one (least-squares slope: 1 = full move, 0 = stays at q_t);
  grasp  on the samples whose demonstrated gripper closes within the executed chunk, the error of that arm's six joints
         at the demonstrated closing step (the grasp pose; joint degrees, not hand position);
  hold   the executed-chunk arm error split by whether that arm's gripper is closed at t (holding) or open;
  copy   per noise time tau, the part of the velocity error that comes from reading the noisy action x_tau (two
         noises, same observation: v1 - v2 against eps1 - eps2, as copy_error.py) next to the full velocity error,
         normalised action units, arm joints only (not comparable between absolute and posture-relative models).
A posture-relative checkpoint (cfg.posture_rel) reads the same samples with its chunks relative to the current joint
vector (its own stats.json); its errors are converted to radians with its per-step scale, so they compare directly.

  python scripts/rt_breakdown.py --ckpt runs/rt_r0/ckpt_60000.pt runs/rt_f2/ckpt_60000.pt [--lang unseen] [--per_task]
      [--no_round none,adapt,head,...]
--lang unseen: every episode reads one of its unseen instructions (drawn once, fixed seed) instead of the seen one it
was trained with, as in the closed-loop eval. --per_task: chunk error per task as well.
"""

import argparse
import contextlib
import glob
import json
import os
import random
import types

import numpy as np
import torch

from rounding_ablation import neurons, smooth_forward
from spikewam.data import Normalizer, SkeletonDataset
from spikewam.text import TextEncoder
from spikewam.train import MODELS

TAUS = (0.05, 0.2, 0.5, 0.8)
ARM = [i for i in range(14) if i not in (6, 13)]       # aloha-agilex: 6 joints + gripper per arm
GRIP = [6, 13]
N_EXEC = 16
TOGGLE = 1.0
BIG_MOVE = 20.0


def sample(model, args, x):
    """Euler integration of the flow from noise x, as sample_actions (with the remainder carried by a SpikeWAM with
    stoch_round, unless its cfg.carry is off)."""
    steps = model.cfg.flow_steps
    with model.carrying() if hasattr(model, "carrying") else contextlib.nullcontext():
        for i in range(steps):
            tau = torch.full((x.shape[0],), 1 - i / steps, device=x.device)
            x = x - model(*args, x, tau)[0].float() / steps
    return x


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--batches", type=int, default=20)
    pa.add_argument("--batch", type=int, default=64)
    pa.add_argument("--lang", default="seen", choices=("seen", "unseen"))
    pa.add_argument("--per_task", action="store_true")
    pa.add_argument("--no_carry", action="store_true", help="stoch_round models: plain rounding at inference")
    pa.add_argument("--no_round", default="none", help="spiking models: comma list of neuron groups whose rounding is "
                    "switched off at inference, one measurement each (scripts/rounding_ablation.py)")
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds = SkeletonDataset(a.data, dev, bench="robotwin")
    gen = torch.Generator(device=dev).manual_seed(0)
    idx = [torch.randint(0, ds.n_steps, (a.batch,), device=dev, generator=gen) for _ in range(a.batches)]
    batches = [ds.get(i) for i in idx]
    noise = [[torch.randn(b["actions"].shape, device=dev, generator=gen) for _ in range(2)] for b in batches]
    languages = ds.languages
    if a.lang == "unseen":                       # same episode order as load_robotwin
        rng = random.Random(0)
        languages = [rng.choice(json.loads(m)["unseen"]) for path in sorted(glob.glob(os.path.join(a.data, "*.npz")))
                     for m in np.load(path, allow_pickle=True)["meta"]]
        assert len(languages) == len(ds.languages)
    suite = [n.split("/")[0] for n in ds.task_names]
    d_ref = SkeletonDataset(a.data, dev, bench="robotwin", proprio=True)     # raw joint vectors and q_t
    big, move, held = [], [], []
    for i in idx:
        held.append(d_ref.act_raw[d_ref.anchor[i]][:, GRIP] < 0.5)                  # (B, 2): gripper closed at t
        k15 = torch.minimum(i + N_EXEC - 1, d_ref.ep_end[i])
        move.append((d_ref.act_raw[k15] - d_ref.act_raw[d_ref.anchor[i]])[:, ARM])
        big.append(move[-1].abs() > np.radians(BIG_MOVE))
    del d_ref
    for ckpt in a.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        if a.no_carry:
            model.cfg.carry = False
        b_model, d_model = batches, ds
        if model.cfg.posture_rel or model.cfg.proprio:   # same samples, posture-relative chunks / joint-angle input
            with open(os.path.join(os.path.dirname(ckpt), "stats.json")) as f:
                d_model = SkeletonDataset(a.data, dev, bench="robotwin", normalizer=Normalizer(json.load(f)),
                                          posture_rel=model.cfg.posture_rel, posture_scale=model.cfg.posture_scale,
                                          proprio=model.cfg.proprio)
            b_model = [d_model.get(i) for i in idx]
        unit = torch.as_tensor(d_model.norm.units(model.cfg.horizon), device=dev)   # radians per normalised unit
        text_all, mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(languages)
        groups = a.no_round.split(",") if hasattr(model, "head_spike") else ["none"]
        for group in groups:
            mods = neurons(model, group)
            for m in mods:
                m.forward = types.MethodType(smooth_forward, m)
            sq_arm = sq_grip = 0.0
            per, by_step, by_joint, spread = {}, 0.0, 0.0, 0.0
            sq_anchor, by_step_anchor, by_joint15 = 0.0, 0.0, 0.0
            delays, never, n_samples, grasp = [], [], 0, []
            sq_big, n_big, sq_rest, n_rest, pd_big, dd_big = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            sq_hold, n_hold, sq_open, n_open = 0.0, 0.0, 0.0, 0.0
            copy = {t: 0.0 for t in TAUS}
            full = {t: 0.0 for t in TAUS}
            for b, (eps1, eps2), bg, mv, hd in zip(b_model, noise, big, move, held):
                args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                        mask_all[b["task"]])
                acts = b["actions"]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for t in TAUS:
                        tau = torch.full((acts.shape[0],), t, device=dev)
                        v1 = model(*args, (1 - t) * acts + t * eps1, tau)[0].float()
                        v2 = model(*args, (1 - t) * acts + t * eps2, tau)[0].float()
                        copy[t] += (((v1 - v2) - (eps1 - eps2)) ** 2 / 2)[..., ARM].mean().item() / len(batches)
                        full[t] += ((v1 - (eps1 - acts)) ** 2)[..., ARM].mean().item() / len(batches)
                    x, x2 = sample(model, args, eps1), sample(model, args, eps2)
                spread = spread + (((x - x2)[:, 0, ARM] * unit[0, ARM]) ** 2).mean() / 2 / len(batches)
                err = (x - acts)[:, :N_EXEC] * unit[:N_EXEC]
                sq_arm += (err[..., ARM] ** 2).mean().item() / len(batches)
                sq_grip += ((x - acts)[:, :N_EXEC, GRIP] ** 2).mean().item() / len(batches)
                by_step = by_step + (err[..., ARM] ** 2).mean((0, 2)) / len(batches)
                by_joint = by_joint + (err[:, 0, ARM] ** 2).mean(0) / len(batches)
                by_joint15 = by_joint15 + (err[:, N_EXEC - 1, ARM] ** 2).mean(0) / len(batches)
                # oracle: the step-0 error removed from every step (how much of the chunk error is a common offset)
                anc = (err[..., ARM] - err[:, :1, ARM]) ** 2
                sq_anchor += anc.mean().item() / len(batches)
                by_step_anchor = by_step_anchor + anc.mean((0, 2)) / len(batches)
                for i, e in zip(b["task"].tolist(), (err[..., ARM] ** 2).mean((1, 2)).tolist()):
                    per.setdefault(suite[i], []).append(e)
                demo, pred = acts[:, :N_EXEC, GRIP].float(), x[:, :N_EXEC, GRIP].float()
                n_samples += demo.shape[0]
                tog = (demo[:, -1] - demo[:, 0]).abs() > TOGGLE                # (B, 2): sample, arm
                mid, side = (demo[:, :1] + demo[:, -1:]) / 2, torch.sign(demo[:, -1:] - demo[:, :1])
                cross = lambda g: torch.where(((g - mid) * side > 0).any(1), ((g - mid) * side > 0).float().argmax(1),
                                              torch.full_like(mid[:, 0], N_EXEC))
                delays += (cross(pred) - cross(demo))[tog].tolist()
                never += (cross(pred) == N_EXEC)[tog].tolist()
                # the arm's joints at the step its gripper closes in the demonstration (the grasp pose)
                closing = tog & (side[:, 0] < 0)
                ea = err[..., ARM].float().view(err.shape[0], err.shape[1], 2, 6)
                at = ea[torch.arange(len(ea), device=dev)[:, None], cross(demo).long().clamp_max(N_EXEC - 1),
                        torch.arange(2, device=dev)[None]]
                grasp += (at ** 2).mean(-1)[closing].tolist()
                arm_sq = (ea ** 2).mean((1, 3))                                     # (B, 2): per sample and arm
                sq_hold, n_hold = sq_hold + (arm_sq * hd).sum().item(), n_hold + hd.sum().item()
                sq_open, n_open = sq_open + (arm_sq * ~hd).sum().item(), n_open + (~hd).sum().item()
                e15 = err[:, N_EXEC - 1, ARM].float() ** 2
                sq_big, n_big = sq_big + (e15 * bg).sum(0), n_big + bg.sum(0)
                sq_rest, n_rest = sq_rest + (e15 * ~bg).sum(0), n_rest + (~bg).sum(0)
                pred_mv = mv + err[:, N_EXEC - 1, ARM].float()        # the sampled step-15 joint minus q_t
                pd_big, dd_big = pd_big + (pred_mv * mv * bg).sum(0), dd_big + (mv * mv * bg).sum(0)
            name = ckpt if group == "none" else f"{ckpt} (no rounding: {group})"
            print(f"{name}: executed chunk RMS arm {sq_arm ** 0.5 * 57.3:.2f} deg, gripper MSE {sq_grip:.4f}")
            print("  arm RMS by chunk step (deg): " + "  ".join(f"{t}: {by_step[t].item() ** 0.5 * 57.3:.2f}"
                                                              for t in (0, 1, 3, 7, 11, 15)))
            print(f"  step-0 error removed from every step (oracle common offset): executed chunk RMS arm "
                  f"{sq_anchor ** 0.5 * 57.3:.2f} deg; by step " + "  ".join(
                      f"{t}: {by_step_anchor[t].item() ** 0.5 * 57.3:.2f}" for t in (1, 3, 7, 11, 15)))
            bias = max(by_step[0].item() - spread.item(), 0)
            print(f"  step 0: spread between samples {spread.item() ** 0.5 * 57.3:.2f} deg (RMS of the difference of "
                  f"two samples / sqrt 2), the rest (bias) {bias ** 0.5 * 57.3:.2f} deg")
            print("  step 0 RMS per joint (deg, left then right arm, joints 1-6): "
                  + " ".join(f"{v ** 0.5 * 57.3:.2f}" for v in by_joint.tolist()))
            print(f"  step {N_EXEC - 1} RMS per joint (deg, left then right arm, joints 1-6): "
                  + " ".join(f"{v ** 0.5 * 57.3:.2f}" for v in by_joint15.tolist()))
            dl = np.array(delays)
            print(f"  gripper toggles ({len(dl)} in {n_samples} samples x 2 arms): crossing step sampled - demonstrated "
                  f"mean {dl.mean():+.2f}, |.| {np.abs(dl).mean():.2f}, late by > 2 steps {np.mean(dl > 2) * 100:.1f}%, "
                  f"early by > 2 {np.mean(dl < -2) * 100:.1f}%, no crossing {np.mean(never) * 100:.1f}%")
            print(f"  grasp pose ({len(grasp)} closings): RMS of the closing arm's joints at the demonstrated closing "
                  f"step {np.mean(grasp) ** 0.5 * 57.3:.2f} deg, median {np.median(grasp) ** 0.5 * 57.3:.2f}, "
                  f"90th percentile {np.percentile(grasp, 90) ** 0.5 * 57.3:.2f}")
            print(f"  executed chunk RMS arm, arms holding (gripper closed at t, "
                  f"{n_hold / (n_hold + n_open) * 100:.0f}% of arms) / open: {(sq_hold / max(n_hold, 1)) ** 0.5 * 57.3:.2f}"
                  f" / {(sq_open / max(n_open, 1)) ** 0.5 * 57.3:.2f} deg")
            rms = lambda sq, n: " ".join(f"{v:.2f}" for v in ((sq / n.clamp_min(1)) ** 0.5 * 57.3).tolist())
            print(f"  step {N_EXEC - 1} RMS per joint, samples whose joint moves > {BIG_MOVE:.0f} deg by then "
                  f"({' '.join(f'{v:.0f}' for v in (n_big / (n_big + n_rest) * 100).tolist())} %): {rms(sq_big, n_big)}")
            print(f"  step {N_EXEC - 1} RMS per joint, the other samples: {rms(sq_rest, n_rest)}")
            print(f"  large moves: sampled / demonstrated move (slope) per joint: "
                  + " ".join(f"{v:.2f}" for v in (pd_big / dd_big).tolist()))
            print("  copy / full velocity error (arm, normalised): "
                  + "  ".join(f"tau {t}: {copy[t]:.4f}/{full[t]:.4f}" for t in TAUS), flush=True)
            if a.per_task:
                print("  per task arm RMS (deg): " + "  ".join(f"{k} {np.mean(v) ** 0.5 * 57.3:.2f}"
                                                             for k, v in sorted(per.items())), flush=True)
            for m in mods:
                del m.forward                 # back to IntSpike.forward


if __name__ == "__main__":
    main()
