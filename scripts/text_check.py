"""Does a frozen text encoder tell the instructions apart the way the policy needs it to? (implementation record 4.12)

No training: nearest-neighbour retrieval with the context the policy would read (pool=mean: the normalised mean of the
token states; cosine after removing the mean over all sentences).
  robotwin       for each episode, one of its unseen instructions (drawn as the closed-loop eval does) is matched to
                 the nearest seen instruction of all tasks (10 per episode, first 10 episodes per task); right if
                 it comes from the same task, and (task, arm) right if that episode also used the same arm ({a}).
  libero-plus    each language-perturbed instruction of LIBERO-Plus is matched to the nearest of the 40 LIBERO
                 instructions; right if it is the one it was rewritten from.

  python scripts/text_check.py [--encoders minilm smoothspike]
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch

from spikewam.text import TextEncoder

RT_DATA = "/mnt/data/user01/spikewam_robotwin/demo_clean"
LP_INSTR = os.path.join(os.environ.get("SW_BENCH", "/home/user01/Minko/spike-new-vla-icml/benchmarks"),
                        "libero_plus_instructions.json")


def embed(enc, texts, bs=256):
    out = torch.cat([enc(texts[i:i + bs])[0][:, 0] for i in range(0, len(texts), bs)])
    return out


def nearest(q, keys):
    mu = torch.cat([q, keys]).mean(0)
    q, keys = [torch.nn.functional.normalize(x - mu, dim=-1) for x in (q, keys)]
    return (q @ keys.T).argmax(1)


def robotwin(enc):
    seen, seen_lab, query, query_lab = [], [], [], []
    rng = random.Random(0)
    for path in sorted(glob.glob(os.path.join(RT_DATA, "*.npz"))):
        task = os.path.basename(path)[:-4]
        for e, m in enumerate(np.load(path, allow_pickle=True)["meta"]):
            m = json.loads(m)
            lab = (task, m["info"]["info"].get("{a}", ""))
            query.append(rng.choice(m["unseen"]))
            query_lab.append(lab)
            if e < 10:
                for s in m["seen"][:10]:
                    seen.append(s)
                    seen_lab.append(lab)
    idx = nearest(embed(enc, query), embed(enc, seen)).tolist()
    task = np.mean([seen_lab[i][0] == q[0] for i, q in zip(idx, query_lab)])
    both = np.mean([seen_lab[i] == q for i, q in zip(idx, query_lab)])
    return f"RoboTwin unseen -> seen: task {100 * task:.1f}%, task+arm {100 * both:.1f}% ({len(query)} queries)"


def libero_plus(enc):
    ins = json.load(open(LP_INSTR))["instructions"]
    lang = {k: v for k, v in ins.items() if "_language_" in k}
    orig = sorted({v for k, v in ins.items() if "_language_" not in k})
    truth = [orig.index(k.split("_language_")[0].replace("_", " ")) for k in lang]
    idx = nearest(embed(enc, list(lang.values())), embed(enc, orig)).tolist()
    return f"LIBERO-Plus language -> LIBERO: {100 * np.mean([i == t for i, t in zip(idx, truth)]):.1f}% " \
           f"({len(lang)} rewrites, {len(orig)} instructions)"


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--encoders", nargs="+", default=["minilm", "smoothspike"])
    a = pa.parse_args()
    dev = torch.device("cuda")
    for name in a.encoders:
        enc = TextEncoder(dev, pool="mean", encoder=name)
        print(f"{name}: {robotwin(enc)}; {libero_plus(enc)}", flush=True)
        tok = TextEncoder(dev, pool="tokens", encoder=name)
        hid, mask = tok(["put the bowl on the plate", "Use the left arm to pick the plastic bottle up."])
        print(f"  token states: RMS per entry {hid[mask].pow(2).mean().sqrt():.2f}, padded tokens read the same "
              f"sentence alike: {torch.allclose(hid[0, :4], tok.model(**tok.tok(['put the bowl on the plate'], return_tensors='pt').to(dev)).last_hidden_state[0, :4].float(), atol=1e-3)}", flush=True)


if __name__ == "__main__":
    main()
