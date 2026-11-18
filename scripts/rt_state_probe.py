"""Where does the spiking model lose the arm posture? (implementation record 4.11)

RoboTwin actions are absolute joint targets, so the policy must know the current joint angles, which the skeleton gives
only through the positions of the arm's link origins, grip site and fingers. For random training frames (not the first
of an episode), a ridge read-out per arm from the 10 arm-node tokens (concatenated) to the arm's 6 current joint angles
(the previous frame's action), at the input (the coordinate adapter's output) and after some blocks (the world
stream; noisy actions at noise time 1). Fit on 80% of the frames, RMSE (degrees) on the rest. The same read-out from
the raw arm-node coordinates is the linear reference.

  python scripts/rt_state_probe.py --ckpt runs/rt_r0/ckpt_60000.pt runs/rt_f2/ckpt_60000.pt [--n 24000]
"""

import argparse

import torch

from spikewam.data import SkeletonDataset
from spikewam.robotwin.layout import ARM_NODES
from spikewam.text import TextEncoder
from spikewam.train import MODELS

JOINTS = ([0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12])          # per arm, grippers (6, 13) left out


def ridge_rmse(x, y, lams=(1e-4, 1e-3, 1e-2)):
    """Fit on the first 80%, RMSE per output column on the rest (the best of a few ridge strengths)."""
    n = int(0.8 * len(x))
    x = torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1).double()
    y = y.double()
    mu, sd = x[:n].mean(0), x[:n].std(0).clamp_min(1e-6)
    sd[-1], mu[-1] = 1, 0
    x = (x - mu) / sd
    xtx, xty, eye = x[:n].T @ x[:n], x[:n].T @ y[:n], torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    errs = [(x[n:] @ torch.linalg.solve(xtx + lam * n * eye, xty) - y[n:]).pow(2).mean(0).sqrt() for lam in lams]
    return torch.stack(errs).min(0).values.float()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--n", type=int, default=24000)
    pa.add_argument("--batch", type=int, default=256)
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds = SkeletonDataset(a.data, dev, bench="robotwin")
    gen = torch.Generator(device=dev).manual_seed(0)
    idx = torch.randint(0, ds.n_steps, (a.n,), device=dev, generator=gen)
    idx = idx[idx > ds.ep_start[idx]]
    half = torch.as_tensor(ds.norm.act_hi - ds.norm.act_lo, device=dev) / 2
    q = ds.act[idx - 1] * half                                   # radians (up to a constant offset)
    cur = ds.skel[idx][:, :2 * ARM_NODES]
    for arm, joints in enumerate(JOINTS):
        err = ridge_rmse(cur[:, arm * ARM_NODES:(arm + 1) * ARM_NODES].flatten(1), q[:, joints])
        print(f"raw coordinates, arm {arm}: " + " ".join(f"{e * 57.3:.2f}" for e in err.tolist()))
    for ckpt in a.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        text_all, mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
        depth = len(model.layers)
        probe_at = sorted({1, depth // 2, depth})
        feats = {0: []} | {l: [] for l in probe_at}
        hooks = [model.layers[l - 1].register_forward_hook(
            lambda m, i, o, l=l: feats[l].append(o[0][:, :2 * ARM_NODES].float().half())) for l in probe_at]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(idx), a.batch):
                b = ds.get(idx[s:s + a.batch])
                ident = model.identity(b["node_id"], b["role"], b["slot"])
                feats[0].append((model.encode_current(b["cur"], ident) - ident)[:, :2 * ARM_NODES].float().half())
                noise = torch.randn(b["actions"].shape, device=dev, generator=gen)
                model(b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                      mask_all[b["task"]], noise, torch.ones(len(noise), device=dev))
        for h in hooks:
            h.remove()
        print(ckpt, flush=True)
        for l, f in feats.items():
            f = torch.cat(f)
            out = []
            for arm, joints in enumerate(JOINTS):
                err = ridge_rmse(f[:, arm * ARM_NODES:(arm + 1) * ARM_NODES].flatten(1).float(), q[:, joints])
                out.append(" ".join(f"{e * 57.3:.2f}" for e in err.tolist()))
            name = "input (coordinate adapter)" if l == 0 else f"after block {l}"
            print(f"  {name}: left {out[0]} | right {out[1]}  (deg, joints 1-6)", flush=True)
        del feats, model


if __name__ == "__main__":
    main()
