"""Where the change-driven events of a spiking image encoder and of SpikeWAM land, per position and per frame.

The frames and the counting are those of scripts/image_events.py --frames and scripts/scene_probe_skeleton.py; this
script keeps what those two sum over, for the event figure of Section 5.3.
  SDT-V3     energy of each cell of the 14 x 14 token grid, per camera and frame; a layer's cost is placed as in
             image_events.Counter.place and carried to the 14 x 14 grid by area, which preserves the sum;
  SpikeWAM   energy of each skeleton node, per frame, action tokens summed.
Both once on the clean scene and once with the noise of the noise table: 1 gray level on every camera frame, 1 cm on
every object keypoint. Runs on the CPU.

  python scripts/event_maps.py --ckpt runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt

With --est the skeletons are the front-end estimates of the same frames (<task>/skel of an npz from
scripts/scene_probe_frontend.py), and --sdt_from reuses the SDT-V3 maps of an earlier run (they do not depend on the
skeleton). 11-15, figure H1 on the front-end skeletons:
  python scripts/event_maps.py --ckpt runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt \
      --est runs/scene_probe_frontend_dr4M_est.npz --sdt_from runs/event_maps.npz --out runs/event_maps_dr4M
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_events as ie  # noqa: E402
from scene_probe_skeleton import E_MAC, E_SOP, TokenEvents  # noqa: E402

from spikewam.events import is_spike, stream_of  # noqa: E402
from spikewam.policy import SkeletonPolicy  # noqa: E402
from spikewam.skeleton import N_ROBOT  # noqa: E402

CELLS = 14
NOISE_GRAY, NOISE_METRE = 1.0, 0.01


class MapCounter(ie.Counter):
    def __init__(self, model):
        super().__init__(model)
        self.map = np.zeros((len(ie.CAMERAS), CELLS, CELLS))

    def place(self, grid, cost, dense, delta):
        super().place(grid, cost, dense, delta)
        if self.region is None:
            return
        g = grid(delta).float()
        cells = F.interpolate(g[None, None], size=(CELLS, CELLS), mode="area")[0, 0] * g.numel() / CELLS ** 2
        self.map[self.cam] += cost * cells.numpy()


class NodeEvents(TokenEvents):
    def reset(self, n):
        self.n, self.prev = n, {}
        self.node, self.action = np.zeros(n), 0.0

    def linear(self, name, mod, x):
        if not self.on or x.dim() != 3:
            return
        x = x.detach().float()[0]
        prev = self.prev.get(name)
        if prev is None or prev.shape != x.shape:
            prev = torch.zeros_like(x)
        if is_spike(x):
            diff, e = (x - prev).abs().sum(-1), E_SOP
        else:
            diff, e = (x != prev).sum(-1).float(), E_MAC
        self.prev[name] = x
        t = (diff * mod.out_features * e).numpy()
        if stream_of(name) == "world":
            self.node += t[:self.n]
        else:
            self.action += float(t.sum())


def run_sdt(paths, noise):
    model = ie.load_sdt(torch.device("cpu"))
    counter = MapCounter(model)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    gen = torch.Generator().manual_seed(0)
    maps, regions = [], []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        rgb, region = z["rgb"], z["region"]
        counter.prev = {}
        for t in range(len(rgb)):
            counter.region, counter.shares = region[t], {}
            counter.map[:] = 0
            before = counter.by_region["delta"].copy()
            for c in range(len(ie.CAMERAS)):
                counter.cam = c
                img = torch.as_tensor(rgb[t, c]).permute(2, 0, 1).float() / 255
                if noise:
                    img = (img + torch.randn(img.shape, generator=gen) * noise / 255).clamp(0, 1)
                img = F.interpolate(img[None], size=224, mode="bilinear", antialias=True)
                with torch.no_grad(), ie.MatmulCount(counter):
                    model.forward_features((img - mean) / std)
            maps.append(counter.map * 1e6)
            regions.append((counter.by_region["delta"] - before) * 1e6)
        print(f"sdt noise={noise} {os.path.basename(path)} {len(maps)} frames {time.strftime('%X')}", flush=True)
    return np.stack(maps), np.stack(regions)


def run_skeleton(paths, ckpt, noise, est=None):
    pol = SkeletonPolicy(ckpt, device="cpu", k=1, flow_steps=1, fixed_noise=True, seed=7)
    model, n_max = pol.model, pol.model.cfg.n_max
    events = NodeEvents(model)
    rng = np.random.default_rng(0)

    class Ext:
        pass

    nodes, actions = [], []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        ext = Ext()
        ext.n_nodes, ext.node_id, ext.role, ext.slot = len(z["node_id"]), z["node_id"], z["role"], z["slot"]
        pol.set_scene(ext, str(z["language"]))
        n = ext.n_nodes
        events.reset(n)
        events.on = True
        for f in range(len(z["t"])):
            events.node[:], events.action = 0, 0.0
            skel = (z["skel"] if est is None else est[os.path.basename(path)[:-4] + "/skel"])[f].astype(np.float32)
            if noise:
                skel[N_ROBOT:] += rng.normal(0.0, noise, (n - N_ROBOT, 3)).astype(np.float32)
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                model.sample_actions(pol.encode(skel), *pol.desc, pol.node_mask, pol.text_tokens, pol.text_mask, k=1, noise=pol.noise)
            row = np.full(n_max, np.nan)
            row[:n] = events.node * 1e6
            nodes.append(row)
            actions.append(events.action * 1e6)
        events.on = False
        print(f"skeleton noise={noise} {os.path.basename(path)} {len(nodes)} frames {time.strftime('%X')}", flush=True)
    return np.stack(nodes), np.array(actions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", default=os.path.join(os.environ.get("SW_DATA", "data"), "scene_frames"))
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_RUNS", "runs"), "event_maps"))
    p.add_argument("--limit", type=int, default=0, help="first tasks only, to time the run")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--est", default="", help="npz with <task>/skel: front-end skeletons of the same frames")
    p.add_argument("--sdt_from", default="", help="npz of an earlier run: reuse its SDT-V3 maps")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    paths = [q for q in sorted(glob.glob(os.path.join(a.frames, "*.npz"))) if not q.endswith("_cams.npz")]
    if a.limit:
        paths = paths[:a.limit]

    est = np.load(a.est) if a.est else None
    nodes, actions = run_skeleton(paths, a.ckpt, 0.0, est)
    nodes_n, actions_n = run_skeleton(paths, a.ckpt, NOISE_METRE, est)
    if a.sdt_from:
        old = np.load(a.sdt_from)
        maps, maps_n, regions, regions_n = old["sdt_map"], old["sdt_map_noise"], old["sdt_region"], old["sdt_region_noise"]
        assert len(maps) == len(nodes), (len(maps), len(nodes))
    else:
        maps, regions = run_sdt(paths, 0.0)
        maps_n, regions_n = run_sdt(paths, NOISE_GRAY)

    index = [(os.path.basename(p)[:-4], f) for p in paths for f in range(len(np.load(p, allow_pickle=True)["t"]))]
    np.savez_compressed(a.out + ".npz", sdt_map=maps, sdt_map_noise=maps_n, sdt_region=regions, sdt_region_noise=regions_n,
                        node_uJ=nodes, node_uJ_noise=nodes_n, action_uJ=actions, action_uJ_noise=actions_n)
    robot, objects = np.nansum(nodes[:, :N_ROBOT], 1).mean(), np.nansum(nodes[:, N_ROBOT:], 1).mean()
    res = {"index": index, "frames": len(index),
           "sdt_uJ_per_plan": float(maps.sum((1, 2, 3)).mean()), "sdt_uJ_per_plan_noise": float(maps_n.sum((1, 2, 3)).mean()),
           "sdt_region_share": dict(zip(ie.REGIONS, (regions.sum((0, 1)) / regions.sum()).tolist())),
           "spikewam_uJ": {"robot_nodes": float(robot), "object_nodes": float(objects), "action_tokens": float(actions.mean())},
           "spikewam_uJ_noise_total": float(np.nansum(nodes_n, 1).mean() + actions_n.mean())}
    print(json.dumps({k: v for k, v in res.items() if k != "index"}, indent=1))
    with open(a.out + ".json", "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
