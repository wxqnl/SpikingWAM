"""Does a policy pull a displaced arm back? (implementation record 4.13)

No training. On fixed RoboTwin training samples, every arm whose gripper is open at t and over the next K steps is
displaced by eps ~ N(0, sigma) per joint, with the input kept consistent: robot nodes + FK(q + eps) - FK(q)
(spikewam/robotwin/fk.py), the proprio channel and the posture-relative anchor read q + eps. The chunk is sampled from
the same flow noise for the clean and the displaced state, and the share of eps still in the commanded joint targets
at chunk step k is reported, (A_k(q + eps) - A_k(q)) . eps / |eps|^2: 1 = the offset is carried along, 0 = the arm is
brought back onto the clean chunk. The samples, eps and flow noise are the same for every checkpoint.

  python scripts/rt_correction.py --ckpt runs/rt_r0_120k/ckpt_120000.pt runs/rt_f2ppt_120k/ckpt_120000.pt
"""

import argparse
import json
import math
import os

import torch

from rt_breakdown import ARM, sample
from spikewam.data import RT_PROPRIO_NODES, Normalizer, SkeletonDataset
from spikewam.text import TextEncoder
from spikewam.train import MODELS

STEPS = (0, 3, 7, 11, 15)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--n", type=int, default=2048)
    pa.add_argument("--batch", type=int, default=64)
    pa.add_argument("--sigma", type=float, default=2.0, help="degrees")
    pa.add_argument("--k_open", type=int, default=8, help="the gripper must stay open this many steps")
    a = pa.parse_args()
    dev = torch.device("cuda")
    ref = SkeletonDataset(a.data, dev, bench="robotwin", proprio=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    i = ref.with_prev[torch.randint(0, len(ref.with_prev), (8 * a.n,), device=dev, generator=gen)]
    q = ref.act_raw[ref.anchor[i]]
    ka = torch.minimum(i[:, None] + torch.arange(a.k_open, device=dev), ref.ep_end[i][:, None])
    open_arm = torch.stack([(q[:, g] > 0.5) & (ref.act_raw[ka, g].min(1).values > 0.5) for g in (6, 13)], 1)
    keep = torch.nonzero(open_arm.any(1))[:, 0][:a.n]
    i, q, open_arm = i[keep], q[keep], open_arm[keep]
    eps = torch.zeros_like(q)
    for arm, sl in enumerate((slice(0, 6), slice(7, 13))):
        eps[:, sl] = torch.randn(len(i), 6, device=dev, generator=gen) * math.radians(a.sigma) * open_arm[:, arm, None]
    noise = torch.randn(len(i), ref.horizon, 14, device=dev, generator=gen)
    del ref
    print(f"{len(i)} samples, {open_arm.float().sum().item():.0f} displaced arms, sigma {a.sigma} deg", flush=True)
    for ckpt in a.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        cfg = model.cfg
        with open(os.path.join(os.path.dirname(ckpt), "stats.json")) as f:
            ds = SkeletonDataset(a.data, dev, bench="robotwin", normalizer=Normalizer(json.load(f)),
                                 posture_rel=cfg.posture_rel, posture_scale=cfg.posture_scale, proprio=True)
        ds.load_fk()
        text_all, mask_all = TextEncoder(dev, pool=cfg.text_pool, encoder=cfg.text_encoder)(ds.languages)
        lo, hi = (torch.as_tensor(v, device=dev) for v in (ds.norm.act_lo, ds.norm.act_hi))
        scale = torch.as_tensor(ds.norm.rel_scale, device=dev) if cfg.posture_rel else None

        def commanded(x, anchor):
            """Absolute joint targets (B, H, 14) in radians from a normalised chunk."""
            if cfg.posture_rel:
                return x * scale + anchor[:, None]
            return (x + 1) / 2 * (hi - lo) + lo

        num, den = torch.zeros(ds.horizon, device=dev), 0.0
        for s in range(0, len(i), a.batch):
            bi, bq, be = i[s:s + a.batch], q[s:s + a.batch], eps[s:s + a.batch]
            b = ds.get(bi)
            pert = b["cur"].clone()
            pert[:, :20, :3] += ds.node_shift(bq, be)
            pert[:, RT_PROPRIO_NODES, -1] = ds.norm.joints(bq + be)
            out = []
            for cur, anchor in ((b["cur"], bq), (pert, bq + be)):
                cur = cur if cfg.proprio else cur[..., :-1]
                args = [cur, b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                        mask_all[b["task"]]]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x = sample(model, args, noise[s:s + a.batch].clone())
                out.append(commanded(x.float(), anchor))
            d = (out[1] - out[0])[..., ARM]
            num += (d * be[:, None, ARM]).sum((0, 2))
            den += (be[:, ARM] ** 2).sum().item()
        g = num / den
        print(f"{ckpt}: share of the displacement kept in the commanded joints, by chunk step: "
              + "  ".join(f"{k}: {g[k].item():.2f}" for k in STEPS), flush=True)


if __name__ == "__main__":
    main()
