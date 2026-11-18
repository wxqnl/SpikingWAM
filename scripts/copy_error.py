"""Can the network carry the noisy action through to its velocity output? (implementation record 4.8)

Flow matching asks for v = eps - a. For one observation and two noises eps1, eps2 at the same tau, the targets
differ by exactly eps1 - eps2, whatever the action; so the error of v1 - v2 against eps1 - eps2 is the part of the
velocity error that comes from reading the noisy action x_tau (the observation-dependent part cancels). Reported
per tau next to the full velocity error, for position / rotation / gripper.

  python scripts/copy_error.py --ckpt runs/r0_ann/ckpt_60000.pt runs/s1_both_D4/ckpt_60000.pt
"""

import argparse

import torch

from loss_breakdown import GROUPS, TAUS, load, setup


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--batches", type=int, default=20)
    pa.add_argument("--batch", type=int, default=64)
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds, batches, noise = setup(a.batches, a.batch)
    for ckpt in a.ckpt:
        model, text_all, text_mask_all = load(ckpt, ds)
        copy = {t: {g: 0.0 for g in GROUPS} for t in TAUS}
        full = {t: {g: 0.0 for g in GROUPS} for t in TAUS}
        for b, (eps1, eps2) in zip(batches, noise):
            args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                    text_mask_all[b["task"]])
            acts = b["actions"]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in TAUS:
                    tau = torch.full((acts.shape[0],), t, device=dev)
                    v1 = model(*args, (1 - t) * acts + t * eps1, tau)[0].float()
                    v2 = model(*args, (1 - t) * acts + t * eps2, tau)[0].float()
                    dc = ((v1 - v2) - (eps1 - eps2)) ** 2 / 2       # per-sample scale, comparable to full
                    df = (v1 - (eps1 - acts)) ** 2
                    for g, s in GROUPS.items():
                        copy[t][g] += dc[..., s].mean().item() / len(batches)
                        full[t][g] += df[..., s].mean().item() / len(batches)
        print(ckpt, "(copy error / full velocity error, pos rot grip)")
        for t in TAUS:
            print(f"  tau {t:4}: " + "  ".join(f"{g} {copy[t][g]:.4f}/{full[t][g]:.4f}" for g in GROUPS), flush=True)


if __name__ == "__main__":
    main()
