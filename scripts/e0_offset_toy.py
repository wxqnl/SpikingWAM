"""E0 (research plan section 5.2): can a spiking network read the offset between two 3D points?

Two points p, q in the LIBERO workspace (robot frame, metres); half of the pairs are near (|q - p| < 5 cm, the
contact regime), half are independent. Each point is coded (direct coordinates or a spike code), embedded by one
linear layer, and a 3-layer MLP on the pair predicts q - p. The spiking variants use integer spikes with D levels
after every linear layer (D = T timesteps at inference); the ANN uses GELU. Reports the median and 95th-percentile
offset error in millimetres on near and far test pairs.
With --relative, the network gets one coded input, the relative vector q - p (computed before any spiking layer),
and the codes are `direct` (q - p in decimetres) and `grid` (grid spike code of q - p).

  python scripts/e0_offset_toy.py [--steps 20000] [--out logs/e0_offset_toy.json]
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikewam.spiking import GrayCode, GridSpikeCode, IntSpike, PlaceCode, VoxelCode

LO = torch.tensor([0.35, -0.35, -0.05])
HI = torch.tensor([0.85, 0.35, 0.35])
CENTER, SCALE = (LO + HI) / 2, (HI - LO) / 2


def sample_pairs(n, device, gen):
    p = LO.to(device) + (HI - LO).to(device) * torch.rand(n, 3, device=device, generator=gen)
    far = LO.to(device) + (HI - LO).to(device) * torch.rand(n, 3, device=device, generator=gen)
    d = torch.randn(n, 3, device=device, generator=gen)
    d = d / d.norm(dim=1, keepdim=True) * 0.05 * torch.rand(n, 1, device=device, generator=gen) ** (1 / 3)
    near = torch.arange(n, device=device) % 2 == 0
    q = torch.where(near[:, None], p + d, far)
    return p, q, near


class Direct(nn.Module):
    dim = 3

    def forward(self, p):
        return (p - CENTER.to(p.device)) / SCALE.to(p.device)


class Decimetres(nn.Module):
    dim = 3

    def forward(self, d):
        return d * 10


class OffsetNet(nn.Module):
    def __init__(self, code, levels, width=256, relative=False):
        super().__init__()
        self.code, self.relative = code, relative
        act = (lambda: IntSpike(levels)) if levels else (lambda: nn.GELU())
        n_in = 1 if relative else 2
        self.embed = nn.Linear(code.dim, width * (2 // n_in))
        self.a0 = act()
        self.mlp = nn.Sequential(nn.Linear(2 * width, 2 * width), act(), nn.Linear(2 * width, 2 * width), act())
        self.out = nn.Linear(2 * width, 3)

    def forward(self, p, q):
        x = self.code(q - p)[:, None] if self.relative else torch.stack([self.code(p), self.code(q)], 1)
        return self.out(self.mlp(self.a0(self.embed(x)).flatten(1)))


def run(code_name, code, levels, steps, device, relative=False):
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(0)
    net = OffsetNet(code, levels, relative=relative).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        p, q, _ = sample_pairs(1024, device, gen)
        loss = F.mse_loss(net(p, q) * 10, (q - p) * 10)  # decimetres
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    test_gen = torch.Generator(device=device).manual_seed(123)
    with torch.no_grad():
        p, q, near = sample_pairs(20000, device, test_gen)
        err = (net(p, q) - (q - p)).norm(dim=1) * 1000
    stat = lambda e: {"median_mm": e.median().item(), "p95_mm": e.quantile(0.95).item()}
    return {"code": code_name, "levels": levels, "relative": relative, "code_dim": code.dim, "near": stat(err[near]),
            "far": stat(err[~near])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--levels", default="0,1,2,4")
    ap.add_argument("--out", default="logs/e0_offset_toy.json")
    ap.add_argument("--relative", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    codes = {"direct": Direct(), "place": PlaceCode(192), "voxel_1cm": VoxelCode(0.01), "voxel_2cm": VoxelCode(0.02),
             "gray_12bit": GrayCode(12), "grid": GridSpikeCode()}
    if a.relative:
        codes = {"rel_direct": Decimetres(), "rel_grid": GridSpikeCode()}
    results = []
    for levels in [int(x) for x in a.levels.split(",")]:
        for name, code in codes.items():
            t0 = time.time()
            r = run(name, code.to(dev), levels, a.steps, dev, a.relative)
            results.append(r)
            print(f"{'ANN' if levels == 0 else f'D={levels}':5s} {name:11s} dim={r['code_dim']:4d}  "
                  f"near median {r['near']['median_mm']:6.2f} p95 {r['near']['p95_mm']:6.2f} mm | "
                  f"far median {r['far']['median_mm']:6.2f} p95 {r['far']['p95_mm']:6.2f} mm  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(a.out, "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
