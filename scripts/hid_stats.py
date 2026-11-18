"""How nonlinear are the spiking neurons in practice? (implementation record 4.8)

On one fixed training batch at noise times tau in {0.2, 0.5, 0.8}, for the neurons in front of Q/K/V (in1), Q, V, the
attention output (o) and the MLP hidden layer (hid), per expert (0 world, 1 action), averaged over layers:
  clipped  share of currents beyond +-D (the only place where a bipolar neuron is not the identity);
  zero     share that rounds to no spike;
  pos/neg  share of positive / negative spikes.

  python scripts/hid_stats.py runs/s1_both_D4/ckpt_60000.pt runs/s1_both_D8/ckpt_60000.pt
"""

import os
import sys

import torch

from spikewam.data import SkeletonDataset
from spikewam.text import TextEncoder
from spikewam.train import MODELS


def main():
    dev = torch.device("cuda")
    ds = SkeletonDataset(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"), dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    b = ds.sample(64, generator=gen)
    for ckpt in sys.argv[1:]:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        text, text_mask = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
        stats = {}

        def hook(name):
            def record(mod, inp, out):
                x = inp[0].float() / mod.log_threshold.exp()
                stats.setdefault(name, []).append((
                    (x.abs() >= mod.levels).float().mean().item(), (x.round() == 0).float().mean().item(),
                    (x > 0.5).float().mean().item(), (x < -0.5).float().mean().item()))
            return record

        for layer in model.layers:
            for ei, e in enumerate(layer.experts):
                for name in ("hid", "in1", "q", "v", "o"):
                    if hasattr(e.n[name], "levels"):
                        e.n[name].register_forward_hook(hook((name, ei)))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in (0.2, 0.5, 0.8):
                tau = torch.full((64,), t, device=dev)
                eps = torch.randn(b["actions"].shape, device=dev, generator=gen)
                model(b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text[b["task"]],
                      text_mask[b["task"]], (1 - t) * b["actions"] + t * eps, tau)
        print(ckpt)
        for (name, ei), v in sorted(stats.items()):
            v = torch.tensor(v).mean(0).tolist()
            print(f"  {name:4s} expert {ei}: clipped {v[0]:.3f}  zero {v[1]:.3f}  pos {v[2]:.3f}  neg {v[3]:.3f}")


if __name__ == "__main__":
    main()
