"""Spiking language encoder (implementation record 4.12): all-MiniLM-L6-v2 with integer spikes before every weight
layer, distilled from the ANN MiniLM on general sentences.

Why not a published spiking BERT: frozen SmoothSpike-BERT (the Spike-driven VLA text encoder) matches an unseen
instruction to its task far less often than MiniLM (scripts/text_check.py: LIBERO-Plus rewrites 51% vs 79%, RoboTwin
unseen 84% vs 91%). Its masked-language pretraining costs 15 points (ANN BERT-base 64%) and its spikes 12 more. Here the
student starts from the weights that already keep paraphrases together and only has to learn to work with spikes.

Each layer (post-LN BERT) with the neurons of SpikeWAM's blocks (spikewam.spiking.IntSpike, D levels):
  attention  in1 -> Q, K, V linear -> q, k, v spikes -> Shiftmax(q k^T / sqrt(d)) v -> o spikes -> dense -> + h, LN
  FFN        in2 -> fc1 -> hid spikes (one-sided, 0..D, the rectification GELU gives) -> fc2 -> + h, LN
Thresholds are learned per channel for the neurons that feed a linear layer (folded into its weight columns at
inference) and per tensor for q and k (whose product is accumulated spike coincidences). The neurons return spikes
times threshold, so the pretrained weights read the scale they were trained on. Kept continuous, as in SmoothSpike and
the SpikeWAM blocks: token embeddings (a table lookup), the residual stream and LayerNorm, the mean over tokens.

Distillation: every layer's hidden states (MSE, per channel normalised by the teacher's spread) plus the sentence
embedding (cosine), on AllNLI and Dolly sentences and the training instructions of LIBERO and RoboTwin (seen ones). No
evaluation instruction (RoboTwin unseen, LIBERO-Plus rewrites) is in the corpus.

  python -m spikewam.spike_text --out /mnt/data/user01/spikewam_text/spike_minilm_d4 [--levels 4] [--steps 20000]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spike_wam import shiftmax
from .spiking import IntSpike

CORPUS = "/mnt/data/user01/text_corpus"
RT_DATA = "/mnt/data/user01/spikewam_robotwin/demo_clean"
LP_INSTR = os.path.join(os.environ.get("SW_BENCH", "/home/user01/Minko/spike-new-vla-icml/benchmarks"),
                        "libero_plus_instructions.json")


class ChannelSpike(IntSpike):
    """IntSpike with a threshold per channel (channels = 1: per tensor); returns spikes times threshold."""

    def __init__(self, levels, bipolar, channels):
        super().__init__(levels, bipolar)
        self.log_threshold = nn.Parameter(torch.zeros(channels))

    def forward(self, current):
        return super().forward(current) * self.log_threshold.exp()


class SpikeBertLayer(nn.Module):
    NEURONS = ("in1", "q", "k", "v", "o", "in2", "hid")

    def __init__(self, layer, levels, attn_bits, heads):
        super().__init__()
        a = layer.attention
        self.q, self.k, self.v = a.self.query, a.self.key, a.self.value
        self.dense, self.ln1 = a.output.dense, a.output.LayerNorm
        self.fc1, self.fc2, self.ln2 = layer.intermediate.dense, layer.output.dense, layer.output.LayerNorm
        d, h = self.q.in_features, self.fc1.out_features
        self.heads, self.bits = heads, attn_bits
        self.n = nn.ModuleDict({"in1": ChannelSpike(levels, True, d), "q": ChannelSpike(levels, True, 1),
                                "k": ChannelSpike(levels, True, 1), "v": ChannelSpike(levels, True, d),
                                "o": ChannelSpike(levels, True, d), "in2": ChannelSpike(levels, True, d),
                                "hid": ChannelSpike(levels, False, h)})

    def split(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, x, bias):
        s = self.n["in1"](x)
        q, k, v = (self.split(self.n[name](lin(s))) for name, lin in (("q", self.q), ("k", self.k), ("v", self.v)))
        a = shiftmax((q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1]) + bias, self.bits)
        o = (a.to(v.dtype) @ v).transpose(1, 2).flatten(2)
        x = self.ln1(x + self.dense(self.n["o"](o)))
        return self.ln2(x + self.fc2(self.n["hid"](self.fc1(self.n["in2"](x)))))


class SpikeMiniLM(nn.Module):
    """forward(input_ids, attention_mask, token_type_ids) -> namespace with last_hidden_state (and hidden_states)."""

    def __init__(self, bert, levels=4, attn_bits=2):
        super().__init__()
        self.embeddings = bert.embeddings
        self.embeddings.dropout = nn.Identity()
        heads = bert.config.num_attention_heads
        self.layers = nn.ModuleList([SpikeBertLayer(l, levels, attn_bits, heads) for l in bert.encoder.layer])
        self.levels, self.attn_bits = levels, attn_bits

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        x = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        bias = torch.zeros(attention_mask.shape, device=x.device).masked_fill(~attention_mask.bool(), float("-inf"))
        bias = bias[:, None, None, :]
        hidden = []
        for layer in self.layers:
            x = layer(x, bias)
            hidden.append(x)
        return types.SimpleNamespace(last_hidden_state=x, hidden_states=hidden)

    def neurons(self):
        return [m for m in self.modules() if isinstance(m, IntSpike)]

    @torch.no_grad()
    def ops(self, input_ids, attention_mask=None, token_type_ids=None, norm_ops=8):
        """(SOP, MAC) of encoding one unpadded instruction (B = 1), counted as spikewam.events does: a spike adds the
        weight column of every output it feeds (Q K^T: each q spike, one per key; A V: each v spike, one per query);
        LayerNorm norm_ops MAC-equivalents per channel and token. The embedding lookup is free."""
        assert input_ids.shape[0] == 1
        for m in self.neurons():
            m.count = []
        self(input_ids, torch.ones_like(input_ids) if attention_mask is None else attention_mask, token_type_ids)
        n, sop, mac = input_ids.shape[1], 0.0, 0.0
        for layer in self.layers:
            d, h = layer.q.in_features, layer.fc1.out_features
            fan = {"in1": 3 * d, "q": n, "v": n, "o": d, "in2": h, "hid": d}
            for name, out in fan.items():
                sop += sum(c[0] for c in layer.n[name].count) * out
            mac += 2 * n * d * norm_ops
        for m in self.neurons():
            m.count = None
        return sop, mac

    @classmethod
    def load(cls, path, device="cpu"):
        from transformers import AutoModel
        from .text import _path
        ck = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=False)
        model = cls(AutoModel.from_pretrained(_path()), **ck["cfg"])
        model.load_state_dict(ck["state"])
        return model.to(device)


@torch.no_grad()
def calibrate(student, batch):
    """Thresholds from the currents the neurons see when the student runs without rounding: 3 spreads over D levels
    (bipolar), or the 99th percentile of the positive part over D (one-sided)."""
    currents = {}
    hooks = [m.register_forward_hook(lambda m, i, o: currents.setdefault(m, i[0].detach().float()))
             for m in student.neurons()]
    for m in student.neurons():
        m.forward_real = m.forward
        m.forward = types.MethodType(lambda self, x: x, m)
    student(**batch)
    for m in student.neurons():
        m.forward = m.forward_real
        del m.forward_real
    for h in hooks:
        h.remove()
    for m, x in currents.items():
        x = x[batch["attention_mask"].bool()]
        if m.log_threshold.numel() == 1:
            x = x.flatten()[:, None]
        if m.bipolar:
            th = 3 * x.std(0)
        else:
            th = torch.quantile(x.clamp_min(0)[:100000], 0.99, dim=0)
        m.log_threshold.copy_((th / m.levels).clamp_min(1e-4).log().reshape(m.log_threshold.shape))


def corpus(seed=0):
    """Training sentences; the evaluation instructions are removed."""
    import pyarrow.parquet as pq
    sents = set()
    nli = pq.read_table(os.path.join(CORPUS, "pair/train-00000-of-00001.parquet"))
    sents.update(nli.column("anchor").to_pylist())
    sents.update(nli.column("positive").to_pylist())
    for line in open(os.path.join(CORPUS, "databricks-dolly-15k.jsonl")):
        r = json.loads(line)
        sents.add(r["instruction"])
        sents.update(s.strip() + "." for s in r["response"].split(". ")[:3] if 3 < len(s.split()) < 40)
    ins = json.load(open(LP_INSTR))["instructions"]
    held_out = {v for k, v in ins.items() if "_language_" in k}
    robot = {v for k, v in ins.items() if "_language_" not in k}
    for path in sorted(glob.glob(os.path.join(RT_DATA, "*.npz"))):
        for m in np.load(path, allow_pickle=True)["meta"]:
            m = json.loads(m)
            robot.update(m["seen"])
            held_out.update(m["unseen"])
    sents = [s for s in sents if s and s not in held_out]
    robot = [s for s in robot if s not in held_out]
    random.Random(seed).shuffle(sents)
    return sents, robot


def main():
    from transformers import AutoModel, AutoTokenizer
    from .text import _path
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", required=True)
    pa.add_argument("--levels", type=int, default=4)
    pa.add_argument("--attn_bits", type=int, default=2)
    pa.add_argument("--steps", type=int, default=20000)
    pa.add_argument("--batch", type=int, default=256)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--robot_frac", type=float, default=0.25, help="share of each batch from robot instructions")
    a = pa.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(_path())
    teacher = AutoModel.from_pretrained(_path()).to(dev).eval().requires_grad_(False)
    student = SpikeMiniLM(AutoModel.from_pretrained(_path()), a.levels, a.attn_bits).to(dev)
    sents, robot = corpus()
    print(f"{len(sents)} general sentences, {len(robot)} robot instructions", flush=True)
    rng = random.Random(0)

    def batch():
        n_robot = int(a.batch * a.robot_frac)
        texts = rng.sample(sents, a.batch - n_robot) + rng.sample(robot, n_robot)
        return tok(texts, padding="longest", truncation=True, max_length=64, return_tensors="pt").to(dev)

    calibrate(student, batch())
    thresholds = [p for n, p in student.named_parameters() if n.endswith("log_threshold")]
    weights = [p for n, p in student.named_parameters() if not n.endswith("log_threshold")]
    opt = torch.optim.AdamW([{"params": weights, "lr": a.lr, "weight_decay": 0.01},
                             {"params": thresholds, "lr": 10 * a.lr, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, 10 * a.lr], total_steps=a.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = batch()
        mask = b["attention_mask"].bool()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t = teacher(**b, output_hidden_states=True).hidden_states[1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = student(**b).hidden_states
        layer = sum(((hs.float() - ht.float()) ** 2)[mask].mean(0).div(ht.float()[mask].var(0) + 1e-3).mean()
                    for hs, ht in zip(s, t)) / len(s)
        pool = lambda h: F.normalize((h.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True), dim=-1)
        sent = (1 - (pool(s[-1]) * pool(t[-1])).sum(-1)).mean()
        loss = layer + 10 * sent
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print(f"step={step} loss={loss.item():.4f} layer={layer.item():.4f} sent_cos={1 - sent.item():.4f} "
                  f"sec={time.time() - t0:.0f}", flush=True)
    torch.save({"cfg": {"levels": a.levels, "attn_bits": a.attn_bits}, "state": student.state_dict(),
                "args": vars(a)}, os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()
