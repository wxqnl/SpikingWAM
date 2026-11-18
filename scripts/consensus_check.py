"""How much of the k-sample gain (Medoid Action Consensus) could an adaptive number of samples keep? (no training)

On fixed LIBERO training samples (EMA weights, inference path) every state gets K candidate chunks (flow_steps Euler
steps from independent noises, as sample_actions). Errors are against the demonstration over the executed part
(steps 0-15), the six motion dimensions, normalised space:
  single   the expected error of one sample (mean over the K candidates);
  medoid-k MAC over k candidates (first h steps, as sample_actions), averaged over the K // k disjoint groups;
  mean-k   the average of k candidates (what integrating candidates over time would give), same groups;
  spread   the MAC distance of the K-medoid to the other candidates, per state (how much the samples disagree), and
           which share of the medoid-3 -> medoid-K gain sits in the states with the largest spread;
  adaptive candidates arrive in groups of 3; stop as soon as the medoid of the n so far is within theta of the
           others (its mean MAC distance), else at K; theta swept over quantiles of the n = 3 medoid distance.
           Reports the mean n (energy ~ skeleton branch + n action branches) and the error of the chosen chunk.

  python scripts/consensus_check.py --ckpt runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt runs/r0_ann/ckpt_60000.pt

With --flow_steps 1 2 3 5 10 every checkpoint is measured once per number of Euler steps on the same states and noises
(one step from pure noise returns the model's estimate of the mean chunk, x - v at tau = 1, which mean-k approximates).
"""

import argparse
import json
import os

import torch

from spikewam.data import SkeletonDataset
from spikewam.policy import retry_oom
from spikewam.text import TextEncoder
from spikewam.train import MODELS

N_EXEC = 16
GROUP = 3
KS = (3, 6, 12, 36)
QUANTILES = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)


def candidates(model, args, k, gen, rows=288):
    """(B, k, H_a, D) chunks from k independent noises per state, as sample_actions before the consensus (integrated
    `rows` sequences at a time, so that the check fits next to other jobs; the noises do not depend on it)."""
    b = args[0].shape[0]
    rep = [x.repeat_interleave(k, 0) for x in args]
    x = torch.randn(b * k, model.cfg.horizon, model.cfg.action_dim, device=args[0].device, generator=gen)
    steps = model.cfg.flow_steps
    out = []
    for r in range(0, b * k, rows):
        xr, ar = x[r:r + rows], [a[r:r + rows] for a in rep]
        for i in range(steps):
            tau = torch.full((xr.shape[0],), 1 - i / steps, device=x.device)
            xr = xr - retry_oom(model, *ar, xr, tau)[0].float() / steps
        out.append(xr)
    return torch.cat(out).view(b, k, model.cfg.horizon, model.cfg.action_dim)


def medoid(x, h, md):
    """Index (B,) of the MAC medoid of x (B, n, H_a, D), as sample_actions, and its mean distance to the others."""
    pre = x[:, :, :h, :md].flatten(2)
    d = (torch.cdist(pre, pre) ** 2 / (h * md)).sum(-1) / (x.shape[1] - 1)
    best = d.argmin(-1)
    return best, d.gather(1, best[:, None])[:, 0]


def measure(model, text_all, mask_all, batches, k_max, h, gen):
    md = model.cfg.motion_dims
    out = {f"{m}-{k}": [] for m in ("medoid", "mean") for k in KS}
    single, dist_n, err_n = [], [], []
    for b in batches:
        args = (b["cur"], b["node_id"], b["role"], b["slot"], b["node_mask"], text_all[b["task"]], mask_all[b["task"]])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = candidates(model, args, k_max, gen)
        demo = b["actions"][:, None, :N_EXEC, :md]
        e = ((x[:, :, :N_EXEC, :md] - demo) ** 2).mean((2, 3))                  # (B, K) error of each candidate
        single.append(e.mean(1))
        for k in KS:
            med, avg = 0.0, 0.0
            for j in range(k_max // k):
                xs = x[:, j * k:(j + 1) * k]
                best, _ = medoid(xs, h, md)
                med = med + e[:, j * k:(j + 1) * k].gather(1, best[:, None])[:, 0]
                avg = avg + ((xs.mean(1)[:, None, :N_EXEC, :md] - demo) ** 2).mean((1, 2, 3))
            out[f"medoid-{k}"].append(med / (k_max // k))
            out[f"mean-{k}"].append(avg / (k_max // k))
        dn, en = [], []
        for n in range(GROUP, k_max + 1, GROUP):                                  # the first n candidates
            best, d = medoid(x[:, :n], h, md)
            dn.append(d)
            en.append(e[:, :n].gather(1, best[:, None])[:, 0])
        dist_n.append(torch.stack(dn, 1))
        err_n.append(torch.stack(en, 1))
    r = {"single": torch.cat(single).mean().item()}
    r.update({key: torch.cat(v).mean().item() for key, v in out.items()})
    dist_n, err_n = torch.cat(dist_n), torch.cat(err_n)                          # (N, K/3)
    spread, gain = dist_n[:, -1], err_n[:, 0] - err_n[:, -1]
    order = spread.argsort(descending=True)
    r["spread_quantiles"] = {str(q): spread.quantile(q).item() for q in (0.1, 0.5, 0.9, 0.99)}
    r["gain_share_top_spread"] = {str(f): (gain[order[:int(f * len(order))]].sum() / gain.sum()).item()
                                  for f in (0.1, 0.2, 0.5)}
    r["adaptive"] = []
    for q in QUANTILES:
        theta = dist_n[:, 0].quantile(q)
        ok = dist_n <= theta
        ok[:, -1] = True
        stop = ok.float().argmax(1)                                               # first n whose medoid is tight
        r["adaptive"].append({"quantile": q, "theta": theta.item(), "mean_n": ((stop + 1) * GROUP).float().mean().item(),
                              "err": err_n.gather(1, stop[:, None]).mean().item()})
    return r


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", nargs="+", required=True)
    pa.add_argument("--batches", type=int, default=40)
    pa.add_argument("--batch", type=int, default=32)
    pa.add_argument("--k", type=int, default=36)
    pa.add_argument("--h", type=int, default=10)
    pa.add_argument("--flow_steps", type=int, nargs="*", default=[], help="Euler steps per sample (default: the ckpt's)")
    pa.add_argument("--out", default=os.path.join(os.environ.get("SW_RUNS", "runs"), "consensus_check.json"))
    a = pa.parse_args()
    dev = torch.device("cuda")
    ds = SkeletonDataset(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"), dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    batches = [ds.sample(a.batch, generator=gen) for _ in range(a.batches)]
    results = {}
    for ckpt in a.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        text_all, mask_all = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)(ds.languages)
        for steps in a.flow_steps or [model.cfg.flow_steps]:
            model.cfg.flow_steps = steps
            key = f"{ckpt} steps={steps}" if a.flow_steps else ckpt
            results[key] = measure(model, text_all, mask_all, batches, a.k, a.h,
                                   torch.Generator(device=dev).manual_seed(1))
            print(key, json.dumps(results[key], indent=1), flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
