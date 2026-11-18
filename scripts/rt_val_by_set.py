"""Held-out error of a RoboTwin front end per set (the training split's held-out episodes), each model in its own label
convention (same-kind order and token reader as it was trained). Prints sets sorted by p90.

  CUDA_VISIBLE_DEVICES=7 python scripts/rt_val_by_set.py runs/perception/rtB4t/ckpt.pt --root $SW_DATA/perception_rt,$SW_DATA/perception_rt2
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spikewam.perception import PerceptConfig, SpikePercept  # noqa: E402
from train_perception import DataRT, model_frame_points, prepare  # noqa: E402

p = argparse.ArgumentParser(); p.add_argument("ckpt"); p.add_argument("--root", required=True); p.add_argument("--out", default="")
a = p.parse_args(); dev = "cuda"
st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
model = SpikePercept(cfg).to(dev).eval(); model.load_state_dict(st["model"])
data = DataRT(a.root, dev, "spikeminilm16", getattr(cfg, "dup_order", False), getattr(cfg, "dup_shape", False))   # 11-15: + dup_shape (label convention as trained)
if getattr(cfg, "text_tokens", False):
    data.load_tokens(dev, "spikeminilm16")
remap = torch.as_tensor([list(cfg.kinds).index(k) if k in cfg.kinds else 0 for k in data.kinds], device=dev)
geo = model_frame_points(data.skel, data.ids, data.roles, data.mask)
if getattr(cfg, "dup_order", False):
    geo[data.dup] = 0.0
rows = {}
for k, name in enumerate(data.names):
    idx = data.val_idx[data.task_np[data.val_idx] == k]
    errs = []
    for kk, images in data.loader(idx, 128, 4):
        i = torch.as_tensor(idx[kk.numpy()], device=dev)
        tok, tm = data.tokens_of(data.task[i], 0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]], geo[i], tok, tm)["fused"].float()
        errs.append((o - data.skel[i]).norm(dim=-1)[data.mask[i]].cpu())
    e = torch.cat(errs) * 100
    rows[name] = (float(e.median()), float(e.quantile(0.9)), int(len(e)))
for name, (p50, p90, n) in sorted(rows.items(), key=lambda r: -r[1][1]):
    print(f"{name:45s} p50 {p50:6.2f} p90 {p90:6.2f} n {n}")
if a.out:
    import json; json.dump(rows, open(a.out, "w"), indent=1)
