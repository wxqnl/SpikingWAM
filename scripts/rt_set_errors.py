"""Per-set error of a RoboTwin front end on a perception cache (every set, up to --per frames each): finds broken sets.

  CUDA_VISIBLE_DEVICES=7 python scripts/rt_set_errors.py runs/perception/rtB4g/ckpt.pt --root $SW_DATA/perception_rtcl
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spikewam.perception import PerceptConfig, SpikePercept
from train_perception import DataRT, model_frame_points, prepare

p = argparse.ArgumentParser(); p.add_argument("ckpt"); p.add_argument("--root", required=True); p.add_argument("--per", type=int, default=60)
a = p.parse_args(); dev = "cuda"
st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
model = SpikePercept(cfg).to(dev).eval(); model.load_state_dict(st["model"])
data = DataRT(a.root, dev, "spikeminilm16")
remap = torch.as_tensor([list(cfg.kinds).index(k) if k in cfg.kinds else 0 for k in data.kinds], device=dev)
geo_all = model_frame_points(data.skel, data.ids, data.roles, data.mask)
rows = []
for k, name in enumerate(data.names):
    idx = np.flatnonzero(data.task_np == k)
    idx = idx[np.linspace(0, len(idx) - 1, min(a.per, len(idx))).astype(int)]
    errs = []
    for kk, images in data.loader(idx, 64, 2):
        i = torch.as_tensor(idx[kk.numpy()], device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], remap[data.kind[i]], geo_all[i] if getattr(cfg, "geo", False) else None)["fused"].float()
        errs.append((o - data.skel[i]).norm(dim=-1)[data.mask[i]].cpu())
    e = torch.cat(errs) * 100
    rows.append((name, float(e.median()), float(e.mean()), float((e > 5).float().mean() * 100)))
rows.sort(key=lambda r: -r[2])
for r in rows:
    print(f"{r[0]:45s} p50 {r[1]:6.2f} mean {r[2]:7.2f} >5cm {r[3]:5.1f}%")
