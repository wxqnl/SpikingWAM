"""D0 probe (research plan section 5.3): how well can a layer's input be read out as the hand-object offset?

For random training steps, the targets are the offsets (metres) from the grip-site node to the top point and to the
centre of the first and second object groups (slots 1, 2). Features are what each layer's weights read: the input of
the world expert's QKV projection at the grip-site token, and the input of the action expert's QKV projection averaged
over the action tokens (noisy actions at noise time 1, the first integration step); for the spiking model these are
spikes. Ridge regression on 80% of the samples, error on the rest (median and 95th percentile, mm).

  python scripts/probe_offsets.py --ckpt runs/<run>/ckpt_60000.pt [--n 30000]
"""

import argparse
import json
import os

import numpy as np
import torch

from spikewam.data import Normalizer, SkeletonDataset
from spikewam.geometry import GRIP_NODE
from spikewam.skeleton import ROLES
from spikewam.text import TextEncoder
from spikewam.train import MODELS


def targets(batch, norm_std, norm_mean):
    """(B, 4, 3) offsets grip -> {slot1 top, slot1 centre, slot2 top, slot2 centre}; mask (B, 4)."""
    pos = batch["cur"] * norm_std + norm_mean
    grip = pos[:, GRIP_NODE]
    out, ok = [], []
    for slot in (1, 2):
        for role in ("top", "center"):
            sel = (batch["slot"] == slot) & (batch["role"] == ROLES[role]) & batch["node_mask"]
            idx = sel.float().argmax(1)
            out.append(pos[torch.arange(len(pos)), idx] - grip)
            ok.append(sel.any(1))
    return torch.stack(out, 1), torch.stack(ok, 1)


def ridge(xtr, ytr, xte, lam=1e-3):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    a = torch.cat([a, torch.ones(len(a), 1, device=a.device)], 1)
    b = torch.cat([b, torch.ones(len(b), 1, device=b.device)], 1)
    w = torch.linalg.solve(a.T @ a + lam * len(a) * torch.eye(a.shape[1], device=a.device), a.T @ ytr)
    return b @ w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", default="ema")
    p.add_argument("--n", type=int, default=30000)
    a = p.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck[a.weights])
    with open(os.path.join(os.path.dirname(a.ckpt), "stats.json")) as f:
        norm = Normalizer(json.load(f))
    ds = SkeletonDataset(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"), dev, normalizer=norm)
    text_all, tmask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
    std, mean = torch.tensor(norm.pos_std, device=dev), torch.tensor(norm.pos_mean, device=dev)

    feats = {}
    hooks = []
    for li, layer in enumerate(model.layers):
        for e, name in ((0, "world_grip"), (1, "action_mean")):
            def hook(mod, inp, li=li, e=e, name=name):
                x = inp[0].float()
                f = x[:, GRIP_NODE] if e == 0 else x.mean(1)
                feats.setdefault(f"L{li}_{name}", []).append(f)
            hooks.append(layer.experts[e].qkv.register_forward_pre_hook(hook))
    hooks.append(model.cur_in.fc2.register_forward_pre_hook(
        lambda mod, inp: feats.setdefault("input_grip", []).append(inp[0].float()[:, GRIP_NODE])))

    gen = torch.Generator(device=dev).manual_seed(0)
    ys, oks = [], []
    with torch.no_grad():
        for _ in range(a.n // 500):
            b = ds.sample(500, generator=gen)
            y, ok = targets(b, std, mean)
            ys.append(y)
            oks.append(ok)
            noise = torch.randn(500, model.cfg.horizon, 7, device=dev, generator=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                      tmask_all[b["task"]], noise, torch.ones(500, device=dev))
    for h in hooks:
        h.remove()
    y, ok = torch.cat(ys), torch.cat(oks)
    names = ["slot1_top", "slot1_center", "slot2_top", "slot2_center"]
    n_tr = int(0.8 * len(y))
    result = {"ckpt": a.ckpt, "n": len(y), "layers": {}}
    for key in ["input_grip"] + [f"L{i}_{n}" for i in range(len(model.layers)) for n in ("world_grip", "action_mean")]:
        x = torch.cat(feats[key])
        row = {}
        for t, tn in enumerate(names):
            m = ok[:, t]
            idx = torch.nonzero(m).squeeze(1)
            tr, te = idx[idx < n_tr], idx[idx >= n_tr]
            pred = ridge(x[tr], y[tr, t], x[te])
            err = (pred - y[te, t]).norm(dim=1) * 1000
            row[tn] = {"median_mm": round(err.median().item(), 2), "p95_mm": round(err.quantile(0.95).item(), 2)}
        result["layers"][key] = row
        print(f"{key:16s} " + "  ".join(f"{tn} {row[tn]['median_mm']:6.2f}/{row[tn]['p95_mm']:6.2f}" for tn in names),
              flush=True)
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    with open(os.path.join(os.path.dirname(a.ckpt), f"probe_{stem}_{a.weights}.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
