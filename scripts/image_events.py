"""Events of a spiking image encoder on the same LIBERO demonstrations, for the skeleton-vs-camera comparison.

The official Spike-driven Transformer V3 (19.0M, ImageNet-pretrained, integer spikes 0..4, the SDT-V3 backbone of
spiking VLAs) encodes both cameras (agentview + wrist, 224 x 224) of the first demonstration of every LIBERO task at
every --stride-th step, the replanning points of energy.py. Counted as in spikewam.events:
  dense  every frame recomputed: a layer with spike input costs FLOPs x mean spike count (SOP), with real input FLOPs
         (MAC; the pixel encoding layer and convolutions fed by a convolution without a spike in between);
  delta  change-driven firing: every neuron sends only the change of its spike count since the previous replanning
         step of the same camera, SOP = FLOPs x mean |change|; real inputs cost FLOPs x share of changed inputs.
Attention products count with the rate of their spike operand (dense) or its change (delta).

With --frames (the renders of scene_frames.py: the same demonstrations and replanning steps, with an instance
segmentation per camera), the energy is also split over regions (background, robot, task objects, other objects): a
layer's cost per input element is its FLOPs / input elements, placed at the element's spatial position (the feature
map of a convolution, the 14 x 14 token grid of the Conv1d MLPs and of the attention products' spike operand); a
position's cost goes to the regions in proportion to their pixels in its cell.

With --pixel_noise s (8-bit levels), every camera image gets independent Gaussian pixel noise of std s / 255 before
resizing (camera sensor noise), so that change-driven firing also sees the frame-to-frame noise of a real camera.

  python scripts/image_events.py [--stride 16] [--episodes 1] [--frames $SW_DATA/scene_frames] [--pixel_noise 2]
"""

import argparse
import glob
import importlib.util
import json
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

SPIKE_VLA = "/home/user01/Minko/spike-new-vla-icml"
SDT_SRC = f"{SPIKE_VLA}/code/third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"
SDT_CKPT = f"{SPIKE_VLA}/pretrained/SDT-V3/V3_19.0M_1x4.pth"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")
LEVELS = 4                        # MultiSpike: round(clamp(x, 0, 4)) / 4
REGIONS = ("background", "robot", "task_objects", "other_objects")
E_SOP, E_MAC = 0.9e-12, 4.6e-12


def spikes(x):
    """Unit-spike counts if x is a MultiSpike output (k / 4, k = 0..4), else None."""
    s = x.float() * LEVELS
    if x.min() >= 0 and s.max() <= LEVELS and torch.all(s == torch.round(s)):
        return s
    return None


def load_sdt(dev):
    spec = importlib.util.spec_from_file_location("sdt_v3_models", SDT_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Efficient_Spiking_Transformer_l()
    ck = torch.load(SDT_CKPT, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not [k for k in missing if not k.startswith("head.")] and not unexpected, (missing, unexpected)
    return model.to(dev).eval()


class Counter:
    def __init__(self, model):
        self.tot = {m: defaultdict(float) for m in ("dense", "delta")}
        self.prev, self.cam = {}, None
        self.region, self.shares = None, {}                   # (cams, H, W) region ids of the current frame
        self.by_region = {m: np.zeros((len(CAMERAS), len(REGIONS))) for m in ("dense", "delta")}   # joules
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                m.register_forward_hook(lambda m, inp, out, name=name: self.layer(name, m, inp[0], out))

    def layer(self, name, m, x, out):
        if isinstance(m, nn.Conv2d):
            flops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            grid = lambda t: t.sum(1)[0]                                  # (1, C, H, W)
        elif isinstance(m, nn.Conv1d):
            flops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]
            grid = lambda t: t.sum(1)[0].view(14, 14)                     # (1, C, 196)
        else:
            flops = x.numel() // m.in_features * m.in_features * m.out_features
            grid = None
        self.account(name, x, flops, grid)

    def account(self, name, x, flops, grid=None):
        key = (self.cam, name)
        prev = self.prev.get(key)
        s = spikes(x)
        if s is not None:
            change = s if prev is None else (s - prev).abs()
            self.tot["dense"]["sop"] += flops * s.mean().item()
            self.tot["delta"]["sop"] += flops * change.mean().item()
            self.prev[key] = s
            self.place(grid, flops / x.numel() * E_SOP, s, change)
        else:
            x = x.float()
            changed = torch.ones_like(x) if prev is None else (x != prev).float()
            self.tot["dense"]["mac"] += flops
            self.tot["delta"]["mac"] += flops * changed.mean().item()
            self.prev[key] = x
            self.place(grid, flops / x.numel() * E_MAC, torch.ones_like(x), changed)

    def place(self, grid, cost, dense, delta):
        if self.region is None:
            return
        assert grid is not None
        for mode, t in (("dense", dense), ("delta", delta)):
            g = grid(t)
            k = (self.cam, tuple(g.shape))
            if k not in self.shares:      # share of each region's pixels in every cell: (regions, h, w)
                one = F.one_hot(torch.as_tensor(self.region[self.cam], device=g.device).long(), len(REGIONS))
                self.shares[k] = F.adaptive_avg_pool2d(one.permute(2, 0, 1)[None].float(), g.shape)[0]
            self.by_region[mode][self.cam] += cost * (self.shares[k] * g[None]).sum((1, 2)).cpu().numpy()


class MatmulCount(TorchFunctionMode):
    """Attention products (q @ k^T, a @ v) inside the SDT-V3 forward."""

    def __init__(self, counter):
        super().__init__()
        self.counter, self.calls = counter, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, "__name__", "") in ("matmul", "__matmul__") and len(args) == 2:
            a, b = args
            if a.dim() >= 3 and b.dim() >= 3:
                flops = out.numel() * a.shape[-1]
                op = a if spikes(a) is not None else b
                grid = (lambda t: t.sum(-1).reshape(-1, 196).sum(0).view(14, 14)) if op.shape[-2] == 196 else None
                self.counter.account(f"matmul{self.calls}", op, flops, grid)
                self.calls += 1
        return out


def frames(path, stride, episodes):
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))[:episodes]
        for d in demos:
            obs = f["data"][d]["obs"]
            n = len(obs[CAMERAS[0]])
            yield {c: obs[c][0:n:stride] for c in CAMERAS}


def rendered(frames_dir):
    """scene_frames.py renders: per task, (images (T, cams, H, W, 3), region maps (T, cams, H, W))."""
    for path in sorted(glob.glob(os.path.join(frames_dir, "*.npz"))):
        z = np.load(path)
        yield os.path.basename(path), z["rgb"], z["region"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default=os.environ.get("SW_DEMOS"))
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--frames", default="", help="scene_frames.py renders (energy also split over regions)")
    p.add_argument("--pixel_noise", type=float, default=0.0, help="std of Gaussian pixel noise, 8-bit levels")
    p.add_argument("--out", default="")
    a = p.parse_args()
    out = a.out or os.path.join(os.environ.get("SW_RUNS", "runs"),
                                "image_events_sdtv3" + ("_regions" if a.frames else "") +
                                (f"_pn{a.pixel_noise:g}" if a.pixel_noise else "") + ".json")
    dev = torch.device("cuda")
    model = load_sdt(dev)
    counter = Counter(model)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    if a.frames:
        episodes = ((name, ([rgb[:, c] for c in range(len(CAMERAS))], region)) for name, rgb, region in rendered(a.frames))
    else:
        episodes = ((os.path.basename(path), ([ep[c] for c in CAMERAS], None))
                    for s in SUITES for path in sorted(glob.glob(os.path.join(a.demos, f"{s}_no_noops_t_256", "*_demo.hdf5")))
                    for ep in frames(path, a.stride, a.episodes))
    gen = torch.Generator(device=dev).manual_seed(0)
    n_replans = 0
    for name, (cams, region) in episodes:
        counter.prev = {}
        for t in range(len(cams[0])):
            if region is not None:
                counter.region, counter.shares = region[t], {}
            for c in range(len(CAMERAS)):
                counter.cam = c
                img = torch.as_tensor(cams[c][t], device=dev).permute(2, 0, 1).float() / 255
                if a.pixel_noise:
                    img = (img + torch.randn(img.shape, device=dev, generator=gen) * a.pixel_noise / 255).clamp(0, 1)
                img = F.interpolate(img[None], size=224, mode="bilinear", antialias=True)
                with torch.no_grad(), MatmulCount(counter):
                    model.forward_features((img - mean) / std)
            n_replans += 1
        print(f"{name[:50]}: {n_replans} replanning steps so far", flush=True)
    res = {"model": "SDT-V3 Efficient_Spiking_Transformer_l 19.0M (1x4)", "cameras": list(CAMERAS),
           "source": a.frames or "demos", "pixel_noise": a.pixel_noise, "stride": a.stride, "episodes_per_task": a.episodes, "n_replans": n_replans}
    for m, t in counter.tot.items():
        sop, mac = t["sop"] / n_replans, t["mac"] / n_replans
        res[m] = {"sop": sop, "mac": mac, "energy_uJ_per_replan": (sop * E_SOP + mac * E_MAC) * 1e6}
        if a.frames:
            uj = counter.by_region[m] / n_replans * 1e6
            res[m]["uJ_by_camera_region"] = uj.tolist()
            res[m]["share"] = dict(zip(REGIONS, (uj.sum(0) / uj.sum()).tolist()))
            res[m]["share_by_camera"] = {c: dict(zip(REGIONS, (u / u.sum()).tolist())) for c, u in zip(CAMERAS, uj)}
    res["delta_over_dense_sop"] = res["delta"]["sop"] / res["dense"]["sop"]
    print(json.dumps(res, indent=1))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
