"""How precisely do the input adapters keep the noisy action? (implementation record 4.8; Spike-VLA 6.2)

Hooks the noisy-action adapter (act_in) and fits a ridge read-out from its hidden layer (spikes for the spiking
model, GELU output for the ANN) and from the adapter's output token back to the noisy action x_tau, per action
dimension group; reports the held-out RMSE in units of the per-dimension std of x_tau. The same read-out from the
coordinate adapter's token (cur_in) back to its input coordinates is reported in mm. Also reports the action errors
of loss_breakdown with the Shiftmax normaliser made exact (no ceil), to see whether the normaliser's steps matter.

  python scripts/input_precision.py --ckpt runs/r0_ann/ckpt_60000.pt runs/s1_both_D4/ckpt_60000.pt
"""

import argparse
import json

import torch

import spikewam.spike_wam as sw
from loss_breakdown import GROUPS, load, measure, setup, show


def ridge_rmse(x, y, lam=1e-3):
    """Fit on the first 80%, report RMSE / std on the rest, per output column."""
    n = int(0.8 * len(x))
    x = torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1).double()
    y = y.double()
    a = x[:n].T @ x[:n] + lam * len(x) * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    w = torch.linalg.solve(a, x[:n].T @ y[:n])
    err = (x[n:] @ w - y[n:]).pow(2).mean(0).sqrt()
    return (err / y[n:].std(0)).float()


class _ExactShiftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, bits):
        live = torch.isfinite(scores).any(-1, keepdim=True)
        maximum = torch.where(live, scores.amax(-1, keepdim=True), torch.zeros_like(scores[..., :1]))
        num = torch.exp2(scores - maximum)
        return num / num.sum(-1, keepdim=True).clamp_min(1)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--batches", type=int, default=20)
    pa.add_argument("--batch", type=int, default=64)
    pa.add_argument("--out", default="runs/input_precision.json")
    pa.add_argument("--exact_shiftmax", action="store_true", help="also the action errors with the exact normaliser")
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds, batches, noise = setup(a.batches, a.batch)
    results = {}
    for ckpt in a.ckpt:
        res = results.setdefault(ckpt, {"x_tau_rmse_over_std": {}, "coordinates_mm": {}})
        model, text_all, text_mask_all = load(ckpt, ds)
        hidden_mod = model.act_in.spike if hasattr(model.act_in, "spike") else None
        for t in (0.2, 0.8):
            rec = {"hidden": [], "token": [], "x": []}
            h1 = (hidden_mod.register_forward_hook(lambda m, i, o: rec["hidden"].append(o.float().flatten(0, 1)))
                  if hidden_mod is not None else
                  model.act_in.fc2.register_forward_pre_hook(lambda m, i: rec["hidden"].append(i[0].float().flatten(0, 1))))
            def keep(m, i, o):
                rec["token"].append(o.float().flatten(0, 1))
                rec["x"].append(i[0].float().flatten(0, 1))
            h2 = model.act_in.register_forward_hook(keep)
            for b, (eps, _) in zip(batches, noise):
                args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                        text_mask_all[b["task"]])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(*args, (1 - t) * b["actions"] + t * eps, torch.full((len(eps),), t, device=dev))
            h1.remove()
            h2.remove()
            x = torch.cat(rec["x"])
            line = []
            for name in ("hidden", "token"):
                r = ridge_rmse(torch.cat(rec[name]), x)
                line.append(f"{name} " + "/".join(f"{r[s].mean():.4f}" for s in GROUPS.values()))
                res["x_tau_rmse_over_std"].setdefault(str(t), {})[name] = {g: float(r[s].mean()) for g, s in GROUPS.items()}
            print(f"{ckpt} tau {t}: x_tau read-out RMSE/std (pos/rot/grip)  " + "  ".join(line), flush=True)
        rec = {"x": [], "token": []}

        def keep_cur(m, i, o):
            rec["x"].append(i[0].float().flatten(0, 1))
            rec["token"].append(o.float().flatten(0, 1))
        hook = model.cur_in.register_forward_hook(keep_cur)
        valid = []
        for b, (eps, _) in zip(batches, noise):
            args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                    text_mask_all[b["task"]])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(*args, b["actions"], torch.full((len(eps),), 0.5, device=dev))
            valid.append(b["node_mask"].flatten())
        hook.remove()
        valid = torch.cat(valid)
        x, tok = torch.cat(rec["x"])[valid], torch.cat(rec["token"])[valid]
        r = ridge_rmse(tok, x)
        std = x[int(0.8 * len(x)):].std(0)
        parts = ["abs", "hand"] if x.shape[1] == 6 else ["abs" if model.cfg.frame == "abs" else "hand"]
        line = []
        for k, part in enumerate(parts):
            sl = slice(3 * k, 3 * k + 3)
            scale = (torch.tensor(model.cfg.pos_std, device=dev) if part == "abs" else torch.full((3,), sw.HAND_SCALE,
                                                                                                    device=dev))
            mm = (r[sl] * std[sl] * scale * 1000).tolist()
            res["coordinates_mm"][part] = mm
            line.append(f"{part} RMSE/std {r[sl].mean():.4f} = " + "/".join(f"{v:.1f}" for v in mm) + " mm (x/y/z)")
        print(f"{ckpt} skeleton coordinates read back from the cur_in token: " + "  ".join(line), flush=True)
        if a.exact_shiftmax and isinstance(model, sw.SpikeWAM):
            orig = sw._Shiftmax
            sw._Shiftmax = _ExactShiftmax
            show(f"{ckpt} with exact Shiftmax normaliser", measure(model, text_all, text_mask_all, batches, noise))
            sw._Shiftmax = orig
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
