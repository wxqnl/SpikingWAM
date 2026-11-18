"""Does the motion input (F3, motion_lag = 1) make the policy continue its last step? (implementation record 4.9)

On fixed training samples (EMA weights, flow_steps Euler steps from the same noise, k = 1):
  turn bins  sampled-chunk error over the executed part (steps 0-15) in quartiles of how much the demonstration
             turns: |mean(a_t..a_t+15) - a_t-1| over the motion dimensions (Q1 smooth .. Q4 turning);
  reliance   how far the sampled chunk moves when the motion input of the robot nodes, or of the object nodes, is
             set to zero (motion models only), against how far it is from the demonstration.

  python scripts/copycat_check.py --ckpt runs/f2_geo_finecur16/ckpt_60000.pt runs/f3_geo_finecur16_motion1/ckpt_60000.pt
"""

import argparse
import os

import torch

from spikewam.data import SkeletonDataset
from spikewam.skeleton import ROBOT_BODIES, ROBOT_SITES, ROBOT_TIPS
from spikewam.text import TextEncoder
from spikewam.train import MODELS

N_ROBOT = len(ROBOT_BODIES) + len(ROBOT_SITES) + len(ROBOT_TIPS)
N_EXEC = 16


def sample_chunk(model, b, cur, text, text_mask, x0):
    x = x0.clone()
    steps = model.cfg.flow_steps
    args = (cur, b["node_id"], b["role"], b["slot"], b["node_mask"], text, text_mask)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(steps):
            tau = torch.full((x.shape[0],), 1 - i / steps, device=x.device)
            v, _ = model(*args, x, tau)
            x = x - v.float() / steps
    return x


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--batches", type=int, default=40)
    pa.add_argument("--batch", type=int, default=64)
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds = SkeletonDataset(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"), dev)
    ds.motion_lag = 1
    gen = torch.Generator(device=dev).manual_seed(0)
    idx = [torch.randint(0, ds.n_steps, (a.batch,), device=dev, generator=gen) for _ in range(a.batches)]
    batches = [ds.get(i) for i in idx]
    noise = [torch.randn(b["actions"].shape, device=dev, generator=gen) for b in batches]
    md = ds.norm.md
    turn = []
    for i, b in zip(idx, batches):
        prev = ds.act[torch.maximum(i - 1, ds.ep_start[i]), :md]
        turn.append((b["actions"][:, :N_EXEC, :md].mean(1) - prev).norm(dim=-1))
    turn = torch.cat(turn)
    edges = torch.quantile(turn, torch.tensor([0.25, 0.5, 0.75], device=dev))
    q = torch.bucketize(turn, edges)
    print("turn quartile edges", [round(e, 3) for e in edges.tolist()])
    for ckpt in a.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        text_all, mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
        motion = model.cfg.motion_lag > 0
        err, shift_r, shift_o = [], [], []
        for b, x0 in zip(batches, noise):
            text, text_mask = text_all[b["task"]], mask_all[b["task"]]
            cur = b["cur"] if motion else b["cur"][..., :3]
            x = sample_chunk(model, b, cur, text, text_mask, x0)
            err.append(((x - b["actions"])[:, :N_EXEC, :md] ** 2).mean((1, 2)))
            if motion:
                for out, sl in ((shift_r, slice(0, N_ROBOT)), (shift_o, slice(N_ROBOT, None))):
                    c = cur.clone()
                    c[:, sl, 3:] = 0
                    y = sample_chunk(model, b, c, text, text_mask, x0)
                    out.append(((y - x)[:, :N_EXEC, :md] ** 2).mean((1, 2)))
        err = torch.cat(err)
        line = f"{ckpt.split('/')[1]:32s} exec MSE {err.mean():.4f} | by turn Q1..Q4 " + " ".join(
            f"{err[q == k].mean():.4f}" for k in range(4))
        if motion:
            sr, so = torch.cat(shift_r), torch.cat(shift_o)
            line += f" | zero robot motion moves the chunk {sr.mean():.4f}, zero object motion {so.mean():.4f}"
            line += " (Q4: " + f"{sr[q == 3].mean():.4f} / {so[q == 3].mean():.4f})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
