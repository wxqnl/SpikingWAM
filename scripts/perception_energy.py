"""Energy of the spiking keypoint front end per plan, counted as scripts/image_events.py counts SDT-V3.

The frames are the renders of scene_frames.py: the first demonstration of every LIBERO task at its replanning steps, both
cameras. A layer with spike input costs FLOPs x mean spike count (SOP), with real input FLOPs (MAC). Change-driven firing
sends only the change of a neuron's spike count since the previous plan of the episode. Attention products count with the
rate of their spike operand. Batch normalization is left out, as for SDT-V3. The output decoding (softmax, expectation,
ray cast and fusion) is counted in full as MAC in both modes.

With --cache the frames are the first demonstration of every task in a perception cache at the replanning stride, with
their real camera matrices and object kinds. A front end whose query reads the camera or the object kind needs this.

  CUDA_VISIBLE_DEVICES=4 python scripts/perception_energy.py --ckpt runs/perception/s160/ckpt.pt [--pixel_noise 1]
  CUDA_VISIBLE_DEVICES=4 python scripts/perception_energy.py --ckpt runs/perception/dr160nf/ckpt.pt --cache $SW_DATA/perception_rs
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from spikewam.perception import BINS_UV, BINS_Z, GRIP, N_ROBOT, TIPS, FrontEnd
from spikewam.text import TextEncoder

E_SOP, E_MAC = 0.9e-12, 4.6e-12
DECODE_FIXED = 60                # ray cast and fusion, per node and view


def is_spike(x):
    x = x.float()
    return bool(x.min() >= 0 and torch.all(x == torch.round(x)))


class Count:
    def __init__(self, model):
        self.tot = {m: defaultdict(float) for m in ("dense", "delta")}
        self.prev, self.rate = {}, defaultdict(list)
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.register_forward_hook(lambda m, inp, out, name=name: self.layer(name, m, inp[0], out))

    def layer(self, name, m, x, out):
        if isinstance(m, nn.Conv2d):
            flops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            flops = x.numel() // m.in_features * m.in_features * m.out_features
        self.account(name, x, flops)

    def account(self, name, x, flops):
        prev = self.prev.get(name)
        if prev is not None and prev.shape != x.shape:
            prev = None
        if is_spike(x):
            s = x.float()
            change = s if prev is None else (s - prev).abs()
            self.tot["dense"]["sop"] += flops * s.mean().item()
            self.tot["delta"]["sop"] += flops * change.mean().item()
            self.rate[name].append(s.mean().item())
            self.prev[name] = s
        else:
            x = x.float()
            changed = torch.ones_like(x) if prev is None else (x != prev).float()
            self.tot["dense"]["mac"] += flops
            self.tot["delta"]["mac"] += flops * changed.mean().item()
            self.prev[name] = x


class MatmulCount(TorchFunctionMode):
    def __init__(self, counter):
        super().__init__()
        self.counter, self.calls = counter, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, "__name__", "") in ("matmul", "__matmul__") and len(args) == 2:
            a, b = args
            if a.dim() >= 3 and b.dim() >= 3:
                self.counter.account(f"matmul{self.calls}", a if is_spike(a) else b, out.numel() * a.shape[-1])
                self.calls += 1
        return out


def scene_episodes(root):
    """(rgb (T, 2, H, W, 3) upright, M or None, skel, language, node_id, role, kinds or None) of scripts/scene_frames.py."""
    for path in sorted(glob.glob(os.path.join(root, "*.npz"))):
        z = np.load(path, allow_pickle=True)
        yield z["rgb"][:, :, :, ::-1], None, z["skel"], str(z["language"]), z["node_id"], z["role"], None


def cache_episodes(root, kinds_from, stride=16):
    """The same from a perception cache: the first demonstration of each task, one frame per replanning interval."""
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        z = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        where = np.flatnonzero(z["ep"] == z["ep"].min())
        where = where[(z["t"][where] - z["t"][where[0]]) % stride == 0]
        rgb = np.load(os.path.join(d, "rgb.npy"), mmap_mode="r")
        name = re.sub(r"_s\d+$", "", os.path.basename(d))
        kinds = [str(k) for k in np.load(sorted(glob.glob(os.path.join(kinds_from, name + "*", "meta.npz")))[0], allow_pickle=True)["kind"]]
        yield np.array(rgb[where]), z["M"][where], z["skel"][where], str(z["language"]), z["node_id"], z["role"], kinds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", default=os.path.join(os.environ.get("SW_DATA", "data"), "scene_frames"))
    p.add_argument("--pixel_noise", type=float, default=0.0, help="std of Gaussian pixel noise, 8-bit levels")
    p.add_argument("--text_encoder", default="spikeminilm16")
    p.add_argument("--cache", default="", help="perception cache root; frames with real camera matrices and object kinds")
    p.add_argument("--kinds_from", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_dr"), help="cache whose meta.npz holds each task's object kinds")
    a = p.parse_args()
    dev = torch.device("cuda")
    fe = FrontEnd(a.ckpt, dev)
    model, size = fe.model, fe.cfg.size
    counter = Count(model)
    enc = TextEncoder(dev, pool="mean", encoder=a.text_encoder)
    gen = torch.Generator(device=dev).manual_seed(0)
    n_plans, nodes_total = 0, 0
    index = {k: i for i, k in enumerate(fe.cfg.kinds)}
    for rgb, mats, skel, language, node_id, role, kinds in (cache_episodes(a.cache, a.kinds_from) if a.cache else scene_episodes(a.frames)):
        text = enc([language])[0][:, 0].float()
        as_long = lambda v: torch.as_tensor(np.asarray(v[N_ROBOT:], np.int64), device=dev)[None]
        ids, role = as_long(node_id), as_long(role)
        kind = torch.as_tensor([index.get(k, 0) for k in kinds], device=dev)[None] if kinds is not None and fe.cfg.kinds else None
        counter.prev = {}
        for t in range(len(rgb)):
            M = torch.as_tensor(mats[t], dtype=torch.float32, device=dev)[None] if mats is not None else torch.eye(3, 4, device=dev)[None, None].repeat(1, 2, 1, 1)
            x = torch.as_tensor(rgb[t].copy(), device=dev).float().permute(0, 3, 1, 2) / 255
            if a.pixel_noise:
                x = (x + torch.randn(x.shape, device=dev, generator=gen) * a.pixel_noise / 255).clamp(0, 1)
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True)
            grip = torch.as_tensor(np.r_[skel[t, GRIP], np.linalg.norm(skel[t, TIPS[0]] - skel[t, TIPS[1]])], dtype=torch.float32, device=dev)[None]
            with torch.no_grad(), MatmulCount(counter):
                model(((x - 0.5) / 0.25)[None], M, text, ids, role, grip, kind)
            n_plans += 1
            nodes_total += ids.shape[1]
    decode = nodes_total * 2 * (2 * (2 * BINS_UV + BINS_Z) + DECODE_FIXED)
    res = {"ckpt": a.ckpt, "size": size, "n_param": sum(p.numel() for p in model.parameters()), "n_plans": n_plans, "pixel_noise": a.pixel_noise,
           "mean_spikes_per_neuron_by_layer": {k: float(np.mean(v)) for k, v in counter.rate.items()}}
    for m, t in counter.tot.items():
        sop, mac = t["sop"] / n_plans, (t["mac"] + decode) / n_plans
        res[m] = {"sop": sop, "mac": mac, "uJ_per_plan": (sop * E_SOP + mac * E_MAC) * 1e6}
    out = os.path.join(os.path.dirname(a.ckpt), "energy" + ("_cache" if a.cache else "") + (f"_pn{a.pixel_noise:g}" if a.pixel_noise else "") + ".json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "mean_spikes_per_neuron_by_layer"}, indent=1))


if __name__ == "__main__":
    main()
