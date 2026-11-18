"""Does a model under-rotate large joint moves by a little on every sample, or skip them on some? (implementation record
4.13)

No training. On fixed RoboTwin training samples whose demonstrated joint moves more than BIG_MOVE degrees from q_t by
chunk step 15, draws n chunks per sample (k = 1 each) and reports, per joint, the sampled move / demonstrated move at
step 15: its spread over the draws of one sample (draw-to-draw), and how the draws split into stay (< 0.3), partial
(0.3-0.8) and full (> 0.8). Amplitude loss puts every draw in "partial"; a skipped move puts some in "stay".

  python scripts/rt_roll_modes.py --ckpt runs/rt_f2_120k/ckpt_20000.pt runs/rt_f2pp_120k/ckpt_20000.pt [--joints 5 11]
"""

import argparse
import json
import os

import numpy as np
import torch

from rt_breakdown import ARM, N_EXEC, sample
from spikewam.data import Normalizer, SkeletonDataset
from spikewam.text import TextEncoder
from spikewam.train import MODELS

BIG_MOVE = 20.0


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--joints", type=int, nargs="+", default=[5, 11], help="indices into the 12 arm joints")
    pa.add_argument("--n", type=int, default=8)
    pa.add_argument("--pool", type=int, default=20000, help="frames searched for large moves")
    a = pa.parse_args()
    dev = torch.device("cuda")
    ref = SkeletonDataset(a.data, dev, bench="robotwin", proprio=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    i = ref.with_prev[torch.randint(0, len(ref.with_prev), (a.pool,), device=dev, generator=gen)]
    move = (ref.act_raw[torch.minimum(i + N_EXEC - 1, ref.ep_end[i])] - ref.act_raw[ref.anchor[i]])[:, ARM]
    sel = {j: torch.nonzero(move[:, j].abs() > np.radians(BIG_MOVE))[:, 0][:256] for j in a.joints}
    del ref
    for ckpt in a.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        with open(os.path.join(os.path.dirname(ckpt), "stats.json")) as f:
            ds = SkeletonDataset(a.data, dev, bench="robotwin", normalizer=Normalizer(json.load(f)),
                                 posture_rel=model.cfg.posture_rel, posture_scale=model.cfg.posture_scale,
                                 proprio=True)
        unit = torch.as_tensor(ds.norm.units(model.cfg.horizon), device=dev)
        text_all, mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
        print(ckpt)
        for j, s in sel.items():
            b = ds.get(i[s])
            if not model.cfg.proprio:
                b["cur"] = b["cur"][..., :-1]
            args = [x.repeat_interleave(a.n, 0) for x in (b["cur"], b["node_id"], b["role"], b["slot"],
                                                          b["node_mask"], text_all[b["task"]], mask_all[b["task"]])]
            acts = b["actions"].repeat_interleave(a.n, 0)
            x = torch.randn(acts.shape, device=dev, generator=gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = sample(model, args, x)
            demo = move[s, j].repeat_interleave(a.n)
            pred = demo + ((x - acts)[:, N_EXEC - 1] * unit[N_EXEC - 1])[:, ARM[j]].float()
            r = (pred / demo).view(-1, a.n)
            print(f"  joint {j % 6 + 1} ({'left' if j < 6 else 'right'}), {len(s)} samples x {a.n} draws: "
                  f"median ratio {r.median():.2f}, draw-to-draw std {r.std(1).median():.2f}; draws stay "
                  f"{(r < 0.3).float().mean() * 100:.0f}%, partial {((r >= 0.3) & (r <= 0.8)).float().mean() * 100:.0f}%, "
                  f"full {(r > 0.8).float().mean() * 100:.0f}%; samples with every draw full "
                  f"{(r > 0.8).all(1).float().mean() * 100:.0f}%, none full {(r <= 0.8).all(1).float().mean() * 100:.0f}%",
                  flush=True)


if __name__ == "__main__":
    main()
