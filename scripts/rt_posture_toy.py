"""Toy check before a RoboTwin run (implementation record 4.11): how precisely can a spiking coordinate adapter keep the
arm posture, with and without each node's offset from its parent in the arm chain?

Each of an arm's 10 nodes is embedded on its own, as the model's coordinate adapter does: Linear -> neurons -> Linear;
a small trunk on the concatenated node tokens (Linear -> neurons, twice) predicts the arm's 6 current joint angles (the
previous frame's action). A variant is input/adapter/trunk:
  input    both: position in the robot frame (normalised) and relative to each grip site (/ HAND_SCALE), as F2;
           chain: both + the offset from the node's parent in the arm chain (/ CHAIN_SCALE; base: 0);
  neurons  s<L>: integer spikes with L levels (threshold 0.5 * 4 / L, as fine_in for L = 16); ann: GELU;
           r<L>: the same spikes, trained with stochastic rounding (floor(x + U(0, 1)), unbiased; straight through)
           and tested three ways: plain rounding, the mean of 16 stochastically rounded passes (what neurons that keep
           their remainder across the 10 flow steps approach), and no rounding.
F2 is both/s16/s4. Held-out RMSE per joint, degrees.
--k K: predict the joint target K steps ahead instead (action[t + K], as step K of the model's chunk; within the
episode). --target rel: predict it relative to the current joint vector q_t = action[t - 1] (normalised by the spread
of the difference), which is added back before the error is measured (posture-relative chunks, record 4.13).

  python scripts/rt_posture_toy.py [--steps 6000] [--arms 1] [--k 15 --target rel] [--variants both/s16/s4 ...]
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikewam.data import SkeletonDataset
from spikewam.robotwin.layout import ARM_NODES, GRIP_NODES
from spikewam.spiking import IntSpike
from spikewam.wam import HAND_SCALE

CHAIN_SCALE = 0.1
PARENT = [0, 0, 1, 2, 3, 4, 5, 6, 7, 7]          # base, links 1-6, grip site, fingers (both on the grip site)
JOINTS = ([0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12])


class StochSpike(IntSpike):
    mode = "stoch"                    # class-wide: "stoch", "round" or "smooth"

    def forward(self, current):
        x = current / self.log_threshold.exp()
        x = x.clamp(-self.levels, self.levels)
        if StochSpike.mode == "smooth":
            return x
        q = torch.floor(x + torch.rand_like(x)) if StochSpike.mode == "stoch" else torch.round(x)
        return x + (q - x).detach()


def neurons(kind):
    if kind == "ann":
        return nn.GELU()
    cls = StochSpike if kind[0] == "r" else IntSpike
    return cls(int(kind[1:]), True, threshold=0.5 * 4 / int(kind[1:]))


class Toy(nn.Module):
    def __init__(self, in_dim, adapter, trunk, hidden=256, tok=128):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(in_dim, hidden), neurons(adapter), nn.Linear(hidden, tok))
        self.trunk = nn.Sequential(nn.Linear(ARM_NODES * tok, 512), neurons(trunk), nn.Linear(512, 512),
                                   neurons(trunk), nn.Linear(512, 6))

    def forward(self, x):
        return self.trunk(self.node(x).flatten(1))


def features(ds, idx, arm, mode):
    cur = ds.skel[idx][:, :2 * ARM_NODES]                                   # normalised
    pos = cur * torch.as_tensor(ds.norm.pos_std, device=cur.device) + torch.as_tensor(ds.norm.pos_mean,
                                                                                      device=cur.device)
    parts = [cur] + [(pos - pos[:, g:g + 1]) / HAND_SCALE for g in GRIP_NODES]
    nodes = slice(arm * ARM_NODES, (arm + 1) * ARM_NODES)
    parts = [p[:, nodes] for p in parts]
    if mode == "chain":
        p = pos[:, nodes]
        parts.append((p - p[:, PARENT]) / CHAIN_SCALE)
    return torch.cat(parts, -1)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--steps", type=int, default=6000)
    pa.add_argument("--batch", type=int, default=1024)
    pa.add_argument("--arms", type=int, nargs="+", default=[0, 1])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--variants", nargs="+", default=["both/s16/s4", "chain/s16/s4", "both/ann/ann"])
    pa.add_argument("--k", type=int, default=-1, help="-1: the current joint vector (action[t - 1])")
    pa.add_argument("--target", default="abs", choices=("abs", "rel"))
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds = SkeletonDataset(a.data, dev, bench="robotwin")
    gen = torch.Generator(device=dev).manual_seed(0)
    idx = torch.arange(ds.n_steps, device=dev)
    idx = idx[idx > ds.ep_start[idx]]
    idx = idx[torch.randperm(len(idx), device=dev, generator=gen)]
    test, train = idx[:20000], idx[20000:]
    half = torch.as_tensor(ds.norm.act_hi - ds.norm.act_lo, device=dev) / 2
    target = lambda i: ds.act[torch.minimum(i + a.k, ds.ep_end[i])]
    base = (lambda i: ds.act[i - 1]) if a.target == "rel" else (lambda i: torch.zeros_like(ds.act[i]))
    scale = (target(train) - base(train)).std(0) if a.target == "rel" else torch.ones_like(half)
    for variant in a.variants:
        mode, adapter, trunk = variant.split("/")
        errs = []
        for arm in a.arms:
            joints = JOINTS[arm]
            torch.manual_seed(a.seed)
            x_te, y_te = features(ds, test, arm, mode), target(test)[:, joints]
            b_te, sc = base(test)[:, joints], scale[joints]
            model = Toy(x_te.shape[-1], adapter, trunk).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps)
            for _ in range(a.steps):
                b = train[torch.randint(0, len(train), (a.batch,), device=dev, generator=gen)]
                loss = F.mse_loss(model(features(ds, b, arm, mode)), (target(b) - base(b))[:, joints] / sc)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            tests = {"": lambda: model(x_te) * sc + b_te}
            if "r" in (adapter[0], trunk[0]):
                tests = {"round ": lambda: model(x_te) * sc + b_te,
                         "mean16 ": lambda: torch.stack([model(x_te) for _ in range(16)]).mean(0) * sc + b_te,
                         "smooth ": lambda: model(x_te) * sc + b_te}
            for name, run in tests.items():
                StochSpike.mode = {"round ": "round", "smooth ": "smooth"}.get(name, "stoch")
                with torch.no_grad():
                    err = ((run() - y_te) * half[joints]).pow(2).mean(0).sqrt()
                errs.append(f"{name}{('left', 'right')[arm]} " + " ".join(f"{e * 57.3:.2f}" for e in err.tolist()))
            StochSpike.mode = "stoch"
        print(f"{variant:16s} k {a.k} {a.target}: {' | '.join(errs)}  (deg, joints 1-6)", flush=True)


if __name__ == "__main__":
    main()
