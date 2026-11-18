"""How far apart does a frozen text encoder put LIBERO instructions that differ in a few words? (implementation record 4.12)

No training. For the 40 LIBERO instructions, with the context the policy reads:
  tokens   pairwise distance of the token states (RMS over the tokens either sentence has, divided by the RMS entry
           of all token states), per suite: the nearest pair and the median pair.
  mean     the same for the sentence vector (pool=mean, scaled to RMS entry 1 as the policy reads it).
  fidelity cosine of each spiking token state with MiniLM's, over all tokens and over the tokens that differ from
           the nearest instruction of the suite (the words that tell the tasks apart).

  python scripts/text_margin.py [--encoders minilm spikeminilm]
"""

import argparse

import numpy as np
import torch

from spikewam.libero_tasks import SUITES, get_suite
from spikewam.text import TextEncoder


def pair_dist(h, m):
    """(N, N) distance of token states h (N, L, d) under masks m (N, L), in units of the RMS entry."""
    rms = h[m].pow(2).mean().sqrt()
    n = len(h)
    d = torch.zeros(n, n)
    for i in range(n):
        both = m[i] | m
        diff = (h[i] - h).pow(2).sum(-1) * both
        d[i] = (diff.sum(1) / both.sum(1) / h.shape[-1]).sqrt() / rms
    return d


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--encoders", nargs="+", default=["minilm", "spikeminilm"])
    a = pa.parse_args()
    dev = torch.device("cuda")
    texts = {s: [t.language for t in get_suite(s).tasks] for s in SUITES}
    allt = sum(texts.values(), [])
    states = {}
    for e in a.encoders:
        tok = TextEncoder(dev, "tokens", e)
        mean = TextEncoder(dev, "mean", e)
        states[e] = (*[x.cpu() for x in tok(allt)], mean(allt)[0][:, 0].cpu())
    i0 = 0
    for s in SUITES:
        n = len(texts[s])
        sl = slice(i0, i0 + n)
        i0 += n
        for e in a.encoders:
            h, m, v = states[e]
            dt = pair_dist(h[sl], m[sl])
            dm = (v[sl, None] - v[None, sl]).pow(2).mean(-1).sqrt()
            off = ~torch.eye(n, dtype=bool)
            print(f"{s:15s} {e:12s} tokens nearest {dt.masked_fill(~off, 9).min(1).values.median():.3f} "
                  f"median {dt[off].median():.3f} | mean nearest {dm.masked_fill(~off, 9).min(1).values.median():.3f} "
                  f"median {dm[off].median():.3f}")
    if "minilm" in states:
        h0, m0, _ = states["minilm"]
        for e in a.encoders:
            if e == "minilm":
                continue
            h, m, _ = states[e]
            cos = torch.nn.functional.cosine_similarity(h, h0, dim=-1)
            # Tokens that differ from the nearest instruction of the suite (by MiniLM token distance).
            ids = TextEncoder(dev, "tokens", "minilm").tok(allt, padding="max_length", truncation=True,
                                                         max_length=h.shape[1], return_tensors="pt")["input_ids"]
            dist = torch.zeros(len(allt), dtype=torch.long)
            i0 = 0
            for s in SUITES:
                n = len(texts[s])
                d = pair_dist(h0[i0:i0 + n], m0[i0:i0 + n]).fill_diagonal_(9)
                dist[i0:i0 + n] = d.argmin(1) + i0
                i0 += n
            differ = (ids != ids[dist]) & m0
            print(f"{e}: token cosine with MiniLM, all tokens {cos[m0].mean():.3f}, "
                  f"distinguishing tokens {cos[differ].mean():.3f} ({differ.sum().item()} tokens)")


if __name__ == "__main__":
    main()
