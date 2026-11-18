"""Where does the action error of a spiking model sit? (research plan 11, gap localisation)

On the same fixed training samples for every checkpoint (EMA weights, inference path: no future tokens):
  velocity  flow-matching action loss at fixed noise times tau (1 = noise), split by action dimension group
            (position, rotation, gripper);
  chunk     error of the sampled chunk (flow_steps Euler steps from noise, k = 1) against the demonstration,
            normalised space, split by dimension group and by chunk part (steps 0-9 executed, 10-31 not), and the
            executed part by suite.

  python scripts/loss_breakdown.py --ckpt runs/r0_ann/ckpt_60000.pt runs/s1_both_D4/ckpt_60000.pt
"""

import argparse
import json
import os

import torch

from spikewam.data import SkeletonDataset
from spikewam.text import TextEncoder
from spikewam.train import MODELS

TAUS = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
GROUPS = {"pos": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}


def setup(batches, batch_size):
    """Fixed samples and noises shared by every checkpoint."""
    dev = torch.device("cuda")
    ds = SkeletonDataset(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"), dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    bs = [ds.sample(batch_size, generator=gen) for _ in range(batches)]
    noise = [(torch.randn(b["actions"].shape, device=dev, generator=gen),
              torch.randn(b["actions"].shape, device=dev, generator=gen)) for b in bs]
    return ds, bs, noise


def load(ckpt, ds):
    dev = torch.device("cuda")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["ema"])
    text_all, text_mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
    return model, text_all, text_mask_all


def measure(model, text_all, text_mask_all, batches, noise, suites):
    """Velocity MSE at fixed tau and sampled-chunk MSE, each split by dimension group (suites: suite of each task)."""
    dev = torch.device("cuda")
    vel = {t: {g: 0.0 for g in GROUPS} for t in TAUS}
    chunk = {part: {g: 0.0 for g in GROUPS} for part in ("exec", "rest")}
    by_suite = {}
    for b, (eps, x0) in zip(batches, noise):
        args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]],
                text_mask_all[b["task"]])
        acts = b["actions"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in TAUS:
                tau = torch.full((acts.shape[0],), t, device=dev)
                v, _ = model(*args, (1 - t) * acts + t * eps, tau)
                err = (v.float() - (eps - acts)) ** 2
                for g, s in GROUPS.items():
                    vel[t][g] += err[..., s].mean().item() / len(batches)
            x = x0.clone()
            steps = model.cfg.flow_steps
            for i in range(steps):
                tau = torch.full((acts.shape[0],), 1 - i / steps, device=dev)
                v, _ = model(*args, x, tau)
                x = x - v.float() / steps
        err = (x - acts) ** 2
        for part, sl in (("exec", slice(0, 10)), ("rest", slice(10, None))):
            for g, s in GROUPS.items():
                chunk[part][g] += err[:, sl, s].mean().item() / len(batches)
        for task, e in zip(b["task"].tolist(), err[:, :10].mean((1, 2)).tolist()):
            by_suite.setdefault(suites[task], []).append(e)
    chunk["exec_by_suite"] = {s: sum(v) / len(v) for s, v in sorted(by_suite.items())}
    return {"velocity": {str(t): v for t, v in vel.items()}, "chunk": chunk}


def show(name, r):
    vel, chunk = r["velocity"], r["chunk"]
    print(name)
    print("  velocity MSE by tau  " + "  ".join(f"{t}: " + "/".join(f"{vel[t][g]:.3f}" for g in GROUPS)
                                           for t in vel) + "   (pos/rot/grip)")
    print("  chunk MSE  exec " + "/".join(f"{chunk['exec'][g]:.4f}" for g in GROUPS)
          + "  rest " + "/".join(f"{chunk['rest'][g]:.4f}" for g in GROUPS), flush=True)
    print("  chunk MSE  exec by suite " + "  ".join(f"{s}: {v:.4f}" for s, v in chunk["exec_by_suite"].items()),
          flush=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--batches", type=int, default=20)
    pa.add_argument("--batch", type=int, default=64)
    pa.add_argument("--out", default=os.path.join(os.environ.get("SW_RUNS", "runs"), "loss_breakdown.json"))
    a = pa.parse_args()
    ds, batches, noise = setup(a.batches, a.batch)
    results = {}
    for ckpt in a.ckpt:
        results[ckpt] = measure(*load(ckpt, ds), batches, noise, [n.split("/")[0] for n in ds.task_names])
        show(ckpt, results[ckpt])
    with open(a.out, "w") as f:
        json.dump({"taus": TAUS, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
