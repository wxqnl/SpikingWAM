"""Change-driven energy of the perception front end split over image regions, as image_events.py splits SDT-V3.

Images and region maps are the scene_frames.py renders (the first demonstration of every LIBERO task at its replanning
steps, both cameras, flipped as in perception_energy.scene_episodes). The camera matrices and object kinds, which only the
queries read, come from the matching frames of the perception cache (same demonstration, one control step later).
Layers are counted as in perception_energy.py. A layer with a spatial input places its cost per input element
(FLOPs / input elements x E_SOP or E_MAC, times the element's spike count or its change) at the element's position on
its feature map, and a position's cost goes to the regions in proportion to their pixels in its cell. Layers without a
spatial input (query projections, the per-node layers of the view heads) and the output decoding form a 'queries' part.

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/perception_regions.py --ckpt runs/perception/dr160XL/ckpt.pt
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception_energy import DECODE_FIXED, E_MAC, E_SOP, is_spike  # noqa: E402
from spikewam.perception import BINS_UV, BINS_Z, GRIP, N_ROBOT, TIPS, FrontEnd  # noqa: E402
from spikewam.text import TextEncoder  # noqa: E402

REGIONS = ("background", "robot", "task_objects", "other_objects")
MODES = ("dense", "delta")


class RegionCount:
    def __init__(self, model, views):
        self.views, self.prev, self.region, self.shares = views, {}, None, {}
        self.by_region, self.queries = None, None                               # joules, on the model's device
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.register_forward_hook(lambda m, inp, out, name=name: self.layer(name, m, inp[0], out))

    def layer(self, name, m, x, out):
        if isinstance(m, nn.Conv2d):         # backbone: the batch holds the views
            flops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            self.account(name, x, flops, lambda t: t.sum(1), range(self.views))
        else:
            self.account(name, x, x.numel() // m.in_features * m.in_features * m.out_features)

    def share(self, v, hw):
        k = (v, hw)
        if k not in self.shares:
            one = F.one_hot(self.region[v].long(), len(REGIONS)).permute(2, 0, 1)[None].float()
            self.shares[k] = F.adaptive_avg_pool2d(one, hw)[0]
        return self.shares[k]

    def account(self, name, x, flops, grid=None, views=None):
        prev = self.prev.get(name)
        if prev is not None and prev.shape != x.shape:
            prev = None
        if is_spike(x):
            s = x.float()
            parts, unit = {"dense": s, "delta": s if prev is None else (s - prev).abs()}, flops / x.numel() * E_SOP
            self.prev[name] = s
        else:
            x = x.float()
            parts = {"dense": torch.ones_like(x), "delta": torch.ones_like(x) if prev is None else (x != prev).float()}
            unit = flops / x.numel() * E_MAC
            self.prev[name] = x
        if self.by_region is None:
            self.by_region = {m: torch.zeros(self.views, len(REGIONS), device=x.device, dtype=torch.float64) for m in MODES}
            self.queries = {m: torch.zeros((), device=x.device, dtype=torch.float64) for m in MODES}
        for mode, t in parts.items():
            if grid is None:
                self.queries[mode] += unit * t.sum()
                continue
            g = grid(t)
            for i, v in enumerate(views):
                self.by_region[mode][v] += unit * (self.share(v, tuple(g.shape[-2:])) * g[i][None]).sum((1, 2))


class MatmulCount(TorchFunctionMode):
    """Attention products of SpikeAttn (batch = views) and the feature-query products of the view heads (one view each)."""

    def __init__(self, counter):
        super().__init__()
        self.counter, self.calls, self.heads = counter, 0, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, "__name__", "") in ("matmul", "__matmul__") and len(args) == 2:
            a, b = args
            if a.dim() >= 3 and b.dim() >= 3:
                name, flops = f"matmul{self.calls}", out.numel() * a.shape[-1]
                self.calls += 1
                op = a if is_spike(a) else b
                side = lambda n: int(round(n ** 0.5))
                if op.dim() == 4:                                  # (views, heads, positions, d)
                    h = side(op.shape[-2])
                    self.counter.account(name, op, flops, lambda t: t.sum((1, 3)).view(t.shape[0], h, h), range(self.counter.views))
                elif op is a:                                      # (1, cells, C) of one view
                    h, v = side(op.shape[-2]), self.heads % self.counter.views
                    self.heads += 1
                    self.counter.account(name, op, flops, lambda t: t.sum(-1).view(1, h, h), [v])
                else:
                    self.counter.account(name, op, flops)
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", default=os.path.join(os.environ.get("SW_DATA", "data"), "scene_frames"))
    p.add_argument("--cache", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rs"))
    p.add_argument("--kinds_from", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_dr"))
    p.add_argument("--text_encoder", default="spikeminilm16")
    p.add_argument("--stride", type=int, default=16)
    a = p.parse_args()
    dev = torch.device("cuda")
    fe = FrontEnd(a.ckpt, dev)
    model, size, views = fe.model, fe.cfg.size, fe.cfg.views
    counter = RegionCount(model, views)
    enc = TextEncoder(dev, pool="mean", encoder=a.text_encoder)
    index = {k: i for i, k in enumerate(fe.cfg.kinds)}
    n_plans, nodes_total, skel_gap = 0, 0, []
    for path in sorted(glob.glob(os.path.join(a.frames, "*.npz"))):
        if path.endswith("_cams.npz"):           # camera matrices of the intervention renders (scene_frames_cams.py)
            continue
        z = np.load(path, allow_pickle=True)
        stem = os.path.basename(path)[:-4]
        d = sorted(glob.glob(os.path.join(a.cache, stem + "*")))[0]
        m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        where = np.flatnonzero(m["ep"] == m["ep"].min())
        where = where[(m["t"][where] - m["t"][where[0]]) % a.stride == 0]
        n = min(len(where), len(z["t"]))           # one task has a frame fewer in the cache
        name = re.sub(r"_s\d+$", "", os.path.basename(d))
        kinds = [str(k) for k in np.load(sorted(glob.glob(os.path.join(a.kinds_from, name + "*", "meta.npz")))[0], allow_pickle=True)["kind"]]
        rgb, region, skel = z["rgb"][:n, :, :, ::-1], z["region"][:n, :, :, ::-1], z["skel"][:n]
        text = enc([str(z["language"])])[0][:, 0].float()
        as_long = lambda v: torch.as_tensor(np.asarray(v[N_ROBOT:], np.int64), device=dev)[None]
        ids, role = as_long(z["node_id"]), as_long(z["role"])
        kind = torch.as_tensor([index.get(k, 0) for k in kinds], device=dev)[None] if fe.cfg.kinds else None
        counter.prev = {}
        for t in range(len(rgb)):
            skel_gap.append(float(np.abs(skel[t] - m["skel"][where[t]]).max()))
            counter.region, counter.shares = torch.as_tensor(region[t].copy(), device=dev), {}
            M = torch.as_tensor(m["M"][where[t]], dtype=torch.float32, device=dev)[None]
            x = torch.as_tensor(rgb[t].copy(), device=dev).float().permute(0, 3, 1, 2) / 255
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True)
            grip = torch.as_tensor(np.r_[skel[t, GRIP], np.linalg.norm(skel[t, TIPS[0]] - skel[t, TIPS[1]])], dtype=torch.float32, device=dev)[None]
            with torch.no_grad(), MatmulCount(counter):
                model(((x - 0.5) / 0.25)[None], M, text, ids, role, grip, kind)
            n_plans += 1
            nodes_total += ids.shape[1]
        print(f"{stem}: {n_plans} plans", flush=True)
    decode = nodes_total * 2 * (2 * (2 * BINS_UV + BINS_Z) + DECODE_FIXED) * E_MAC
    res = {"ckpt": a.ckpt, "frames": a.frames, "cache": a.cache, "n_plans": n_plans,
           "skel_gap_cm": {"median": float(np.median(skel_gap)) * 100, "max": float(np.max(skel_gap)) * 100}}
    for mode in MODES:
        uj = counter.by_region[mode].cpu().numpy() / n_plans * 1e6
        q = (counter.queries[mode].item() + decode) / n_plans * 1e6
        total = uj.sum() + q
        res[mode] = {"uJ_per_plan": total, "uJ_by_camera_region": uj.tolist(), "uJ_queries": q,
                     "share": {**dict(zip(REGIONS, (uj.sum(0) / total).tolist())), "queries": q / total},
                     "share_of_image_part": dict(zip(REGIONS, (uj.sum(0) / uj.sum()).tolist()))}
    out = os.path.join(os.path.dirname(a.ckpt), "energy_regions.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
