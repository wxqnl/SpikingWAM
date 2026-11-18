"""Skeleton-vs-camera probes, camera side: a LIBERO image policy of the Spike-VLA code base on the scenes of
scene_frames.py (spiking Spike-Driven VLA base with its LIBERO-trained SDT-V3, or the ANN TurboVLA).

The policy is built by the Spike-VLA evaluator (as its analysis scripts do), so preprocessing, normalisation and
weights are those of its LIBERO runs. Per frame, one batch holds the frame and all its interventions:
  actions    the predicted env action chunk of every render (12 x 7);
  attention  the action head's cross-attention weight on every visual token (attend_probe.AttendTaps: spiking heads
             renormalised per row; averaged over layers, heads, queries and timesteps), split over the regions of
             scene_frames.py by the share of each region's pixels in the token's cell; also each region's share of
             the image area.
With --events (spiking SDT-V3 only), every frame is also encoded alone, in demo order, and the vision encoder's
events are split over regions: a layer's cost per input element is its FLOPs / input elements; a spike input costs
that per unit spike (dense) or per unit of change since the previous frame of the same task and camera (delta), at
the input's spatial position (14 x 14 tokens for the Conv1d MLPs, the token axis for attention products); real
inputs (the pixel encoding) cost MACs, all of them (dense) or those whose input changed (delta). A position's cost
goes to the regions in proportion to their pixels in its cell.

  source <scratchpad>/vla_args.sh; cd $SPIKE_CODE_ROOT
  python <spikewam>/scripts/scene_probe_image.py $(vla_args $BASE ema <log>) --frames <dir> --out <file> [--events]
"""

import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

import attend_probe
import reliance_policy
from turbovla.evaluation.policy import normalized_action_to_env_action, quat2axisangle

REGIONS = ("background", "robot", "task_objects", "other_objects")
LEVELS = 4
E_SOP, E_MAC = 0.9e-12, 4.6e-12


def spikes(x):
    s = x.float() * LEVELS
    if x.min() >= 0 and s.max() <= LEVELS and torch.all(s == torch.round(s)):
        return s
    return None


def region_share(region, hw):
    """(cams, 256, 256) region ids -> (cams, 4, h, w) share of each region's pixels in every cell."""
    one = F.one_hot(torch.as_tensor(region).long(), len(REGIONS)).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(one, hw)


class RegionEvents:
    """Vision-encoder events split over regions (see the module docstring)."""

    def __init__(self, encoder):
        self.on, self.inside, self.view, self.region, self.prev = False, False, -1, None, {}
        self.tot = {m: {"sop": np.zeros((2, len(REGIONS))), "mac": np.zeros((2, len(REGIONS)))}
                    for m in ("dense", "delta")}
        self.unplaced = {"dense": 0.0, "delta": 0.0}
        first = None
        for name, m in encoder.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                first = first or m
                m.register_forward_hook(lambda m, inp, out, name=name: self.layer(name, m, inp[0], out))
        first.register_forward_pre_hook(lambda m, inp: setattr(self, "view", self.view + 1))
        encoder.register_forward_pre_hook(lambda m, inp: setattr(self, "inside", True))
        encoder.register_forward_hook(lambda m, inp, out: setattr(self, "inside", False))

    def layer(self, name, m, x, out):
        if not self.on:
            return
        if isinstance(m, nn.Conv2d):
            flops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Conv1d):
            flops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]
        else:
            flops = x.numel() // m.in_features * m.in_features * m.out_features
        if x.dim() == 4:
            grid = lambda t: t.sum(1)[0]                          # (1, C, H, W) -> (H, W)
        elif x.dim() == 3 and x.shape[-1] == 196:
            grid = lambda t: t.sum(1)[0].view(14, 14)             # (1, C, 196)
        else:
            grid = None
        self.account(name, x, flops, grid)

    def account(self, name, x, flops, grid):
        key, cam = (self.view % 2, name), self.view % 2
        prev = self.prev.get(key)
        cost = flops / x.numel()
        s = spikes(x)
        if s is not None:
            change = s if prev is None else (s - prev).abs()
            self.prev[key] = s
            for mode, t in (("dense", s), ("delta", change)):
                self.place(mode, "sop", cam, t, cost, grid)
        else:
            x = x.float()
            changed = torch.ones_like(x) if prev is None else (x != prev).float()
            self.prev[key] = x
            for mode, t in (("dense", torch.ones_like(x)), ("delta", changed)):
                self.place(mode, "mac", cam, t, cost, grid)

    def place(self, mode, kind, cam, t, cost, grid):
        if grid is None:
            self.unplaced[mode] += cost * t.sum().item() * (E_SOP if kind == "sop" else E_MAC)
            return
        g = grid(t)
        share = region_share(self.region[cam:cam + 1], tuple(g.shape))[0].to(g.device)      # (4, h, w)
        self.tot[mode][kind][cam] += cost * (share * g[None]).sum((1, 2)).cpu().numpy()


class MatmulEvents(TorchFunctionMode):
    """Attention products inside the vision encoder, placed by the token axis of their spike operand."""

    def __init__(self, counter):
        super().__init__()
        self.counter, self.calls, self.view = counter, 0, None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.counter.on and self.counter.inside and getattr(func, "__name__", "") in ("matmul", "__matmul__") and len(args) == 2:
            a, b = args
            if a.dim() >= 3 and b.dim() >= 3:
                if self.view != self.counter.view:            # products are numbered within each view's encoding
                    self.view, self.calls = self.counter.view, 0
                flops = out.numel() * a.shape[-1]
                op = a if spikes(a) is not None else b
                if op.shape[-2] == 196:
                    grid = lambda t: t.sum(-1).reshape(-1, 196).sum(0).view(14, 14)
                elif op.shape[-1] == 196:
                    grid = lambda t: t.reshape(-1, 196).sum(0).view(14, 14)
                else:
                    grid = None
                self.counter.account(f"matmul{self.calls}", op, flops, grid)
                self.calls += 1
        return out


def robot_state8(rs):
    """scene_frames robot_state (eef pos, quaternion, gripper qpos) -> the 8-dim state of the LIBERO policies."""
    return np.concatenate([rs[:3], quat2axisangle(rs[3:7]), rs[7:9]]).astype(np.float32)


def main():
    argv = sys.argv[1:]
    opts = {"--frames": None, "--out": None}
    for flag in list(opts):
        i = argv.index(flag)
        opts[flag] = argv[i + 1]
        del argv[i:i + 2]
    events = "--events" in argv
    if events:
        argv.remove("--events")
    sys.argv = [sys.argv[0], *argv]
    policy = reliance_policy.build_policy()
    model = policy.model
    taps = attend_probe.AttendTaps(model)
    counter = RegionEvents(model.vision_encoder) if events else None
    rows = {"actions": [], "same_actions": [], "iv_actions": [], "attn": [], "iv_attn": [], "area": []}
    meta = []
    for path in sorted(glob.glob(os.path.join(opts["--frames"], "*.npz"))):
        z = np.load(path)
        lang = str(z["language"])
        if counter is not None:                       # events: frames alone, in demo order
            counter.prev, counter.on = {}, True
            for f in range(len(z["t"])):
                counter.region, counter.view = z["region"][f], -1
                with MatmulEvents(counter):
                    policy.predict_normalized_action_chunk(z["rgb"][f, 0], z["rgb"][f, 1], lang,
                                                           robot_state8(z["robot_state"][f]))
            counter.on = False
        for f in range(len(z["t"])):
            ivs = np.flatnonzero(z["iv_frame"] == f)
            # the frame twice: its second copy measures what the batch alone changes (the null intervention)
            prim = [z["rgb"][f, 0]] * 2 + [z["iv_rgb"][i, 0] for i in ivs]
            wrist = [z["rgb"][f, 1]] * 2 + [z["iv_rgb"][i, 1] for i in ivs]
            states = [robot_state8(z["robot_state"][f])] * 2 + [robot_state8(z["iv_robot_state"][i]) for i in ivs]
            pred, feats = taps.run(policy, prim, wrist, states, lang)
            env = np.stack([[normalized_action_to_env_action(r) for r in chunk] for chunk in pred])
            attn = feats["attn_map"].astype(np.float32).mean(1)                    # (B, 392) camera-major
            g = int(round((attn.shape[1] / 2) ** 0.5))                              # 14 (SDT-V3), 16 (DINOv3)
            share = region_share(z["region"][f], (g, g)).flatten(2).permute(0, 2, 1).reshape(-1, 4).numpy()
            reg = attn @ share                                                      # (B, 4)
            rows["actions"].append(env[0])
            rows["same_actions"].append(env[1])
            rows["attn"].append(reg[0])
            rows["iv_actions"] += list(env[2:])
            rows["iv_attn"] += list(reg[2:])
            rows["area"].append(share.mean(0))
            meta.append({"task": os.path.basename(path)[:-4], "frame": f, "t": int(z["t"][f]),
                         "closed": bool(z["closed"][f]),
                         "iv": [{"kind": str(z["iv_kind"][i]), "object": str(z["iv_object"][i])} for i in ivs]})
        print(path, len(meta), flush=True)
    out = opts["--out"]
    np.savez_compressed(out + ".npz", **{k: np.stack(v) for k, v in rows.items()})
    res = {"frames": len(meta), "meta": meta}
    if counter is not None:
        n = len(meta)
        res["events"] = {m: {"sop_by_cam_region": (t["sop"] / n).tolist(), "mac_by_cam_region": (t["mac"] / n).tolist(),
                             "energy_uJ_by_cam_region": ((t["sop"] * E_SOP + t["mac"] * E_MAC) / n * 1e6).tolist(),
                             "unplaced_uJ": counter.unplaced[m] / n * 1e6}
                         for m, t in counter.tot.items()}
    with open(out + ".json", "w") as f:
        json.dump(res, f)
    print("wrote", out)


if __name__ == "__main__":
    main()
