"""How focused is the action tokens' attention on the skeleton (research plan 4.10)?

Along the first --episodes demonstrations of the chosen tasks, the policy samples action chunks every --stride steps
(as in closed-loop control). For every self-attention call, every head and every action-token query, the attention
mass on the current-skeleton nodes is renormalised over those nodes, and we report per layer:
  nodes   share of the query's attention that goes to skeleton nodes;
  actions share that goes to the action tokens themselves (the rest goes to future tokens);
  top1    largest renormalised weight on one node (1 = one node only);
  eff     effective number of attended nodes, exp(entropy) of the renormalised weights.
The ANN's weights are softmax (recomputed from its queries and keys); the spiking model's are Shiftmax, or the
winners in competitive-selection heads, reported apart ("wta") from the other heads ("rest") and together ("all").

  python scripts/attention_focus.py --ckpt runs/r0_ann/ckpt_60000.pt runs/s1_both_D4/ckpt_60000.pt \
      --tasks libero_10/04,libero_10/06
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from spikewam.data import Normalizer
from spikewam.text import TextEncoder
from spikewam.train import MODELS
from spikewam.wam import modulate


class Recorder:
    def __init__(self, model):
        self.model, self.rows = model, []
        self.n_nodes = 0
        for li, layer in enumerate(model.layers):
            if hasattr(layer, "attend"):
                orig = layer.attend

                def attend(q, k, v, bias, li=li, layer=layer, orig=orig, **kw):
                    out = orig(q, k, v, bias, **kw)
                    if kw.get("select"):
                        from spikewam.spike_wam import shiftmax
                        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias
                        a = shiftmax(scores, layer.bits)
                        if layer.wta_heads:
                            h = layer.wta_heads
                            self.add(li, layer.weights[:, :h], "wta")
                            self.add(li, a[:, h:], "rest")
                        self.add(li, a if not layer.wta_heads else torch.cat([layer.weights[:, :h], a[:, h:]], 1))
                    return out
                layer.attend = attend
            else:
                layer.register_forward_pre_hook(lambda m, args, li=li: self.ann(li, m, args))

    def ann(self, li, layer, args):
        xs, mods, mask, _, _ = args
        m = [mod + e.mod_table for mod, e in zip(mods, layer.experts)]
        qkv = [e.qkv(modulate(e.norm1(x), mm[:, :, 0], mm[:, :, 1])) for x, e, mm in zip(xs, layer.experts, m)]
        q, k = (torch.cat([layer.split(t.chunk(3, -1)[j]) for t in qkv], 2) for j in range(2))
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        self.add(li, scores.softmax(-1))

    def add(self, li, a, group="all"):
        """a: (B, H, L, L) weights over [nodes (N_MAX, padded), future tokens, action tokens]."""
        act = a[:, :, -self.n_act:].float()                       # action-token queries
        nodes = act[..., :self.n_nodes]
        mass = nodes.sum(-1)
        p = nodes / mass[..., None].clamp_min(1e-9)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)
        self_mass = act[..., -self.n_act:].sum(-1)               # attention on the action tokens themselves
        self.rows.append((group, li, mass.mean().item(), p.amax(-1).mean().item(), ent.exp().mean().item(),
                          self_mass.mean().item()))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--tasks", default="libero_10/04,libero_10/06,libero_10/00,libero_10/08")
    pa.add_argument("--episodes", type=int, default=3)
    pa.add_argument("--stride", type=int, default=16)
    a = pa.parse_args()
    dev = torch.device("cuda")
    root = os.path.join(os.environ.get("SW_DATA", "data"), "skeletons")
    results = {}
    for ckpt in a.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        with open(os.path.join(os.path.dirname(ckpt), "stats.json")) as f:
            norm = Normalizer(json.load(f))
        text = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)
        rec = Recorder(model)
        rec.n_act = model.cfg.horizon
        gen = torch.Generator(device=dev).manual_seed(0)
        for task in a.tasks.split(","):
            d = np.load(os.path.join(root, task + ".npz"), allow_pickle=True)
            n = d["skel"].shape[1]
            rec.n_nodes = n
            desc = [torch.as_tensor(d[k], device=dev)[None].long() for k in ("node_id", "role", "slot")]
            mask = torch.ones(1, n, dtype=torch.bool, device=dev)
            tt, tm = text([str(d["language"])])
            for start, length in list(zip(d["ep_start"], d["ep_len"]))[:a.episodes]:
                for t in range(0, int(length), a.stride):
                    cur = torch.as_tensor(norm.pos(d["skel"][start + t].astype(np.float32)), device=dev)[None]
                    with torch.no_grad():
                        model.sample_actions(cur, *desc, mask, tt, tm, k=1, generator=gen)
        results[ckpt] = {}
        print(ckpt, "(layer: nodes share, top1, effective nodes, share on action tokens)")
        for group in dict.fromkeys(r[0] for r in rec.rows):
            rows = np.array([r[1:] for r in rec.rows if r[0] == group])
            per_layer = {int(li): rows[rows[:, 0] == li, 1:].mean(0).round(3).tolist()
                         for li in np.unique(rows[:, 0])}
            results[ckpt][group] = per_layer
            print(f" heads: {group}")
            for li, (m, t1, eff, sm) in per_layer.items():
                print(f"  L{li}  nodes {m:.3f}  top1 {t1:.3f}  eff {eff:5.2f}  actions {sm:.3f}")
            allr = rows[:, 1:].mean(0)
            print(f"  all nodes {allr[0]:.3f}  top1 {allr[1]:.3f}  eff {allr[2]:5.2f}  actions {allr[3]:.3f}",
                  flush=True)
    with open(os.path.join(os.environ.get("SW_RUNS", "runs"), "attention_focus.json"), "w") as f:
        json.dump({"tasks": a.tasks, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
