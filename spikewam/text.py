"""Frozen language encoder, run once per instruction and cached.

encoder="minilm": all-MiniLM-L6-v2 (an ANN; SkeleWAM does not name its encoder; trained on paraphrase pairs, which
LIBERO-Plus's rewritten instructions need).
encoder="smoothspike": SmoothSpike-BERT (BERT-base with LIF neurons, T = 4; the fused official checkpoint, as the
Spike-driven VLA text encoder; spike-driven attention and FFN, the residual stream and LayerNorm stay continuous).
encoder="spikeminilm" / "spikeminilm16": MiniLM with integer spikes (4 / 16 levels) before every weight layer, distilled
from MiniLM (spikewam/spike_text.py; checkpoints spike_minilm_d4 / spike_minilm_d16 in SW_SPIKE_TEXT).
encoder="bert": bert-base-uncased (an ANN, masked-language pretraining as SmoothSpike's; for scripts/text_check.py).

pool="tokens": the token states (up to MAX_TOKENS context tokens). pool="mean": the mean over tokens, L2-normalised,
scaled by sqrt(dim) to unit variance per entry, as a single context token; for MiniLM it is the sentence embedding it
was trained to keep invariant under paraphrase (R0 with token states: 68% on LIBERO-Plus language).
"""

from __future__ import annotations

import glob
import os
import sys

import torch

MINILM = os.environ.get("SW_MINILM", os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"))
SMOOTHSPIKE = os.environ.get("SW_SMOOTHSPIKE",
                             "/home/user01/Minko/spike-new-vla-icml/pretrained/SmoothSpike/smoothspike-bert-base-fused")
SPIKE_TEXT = os.environ.get("SW_SPIKE_TEXT", "/mnt/data/user01/spikewam_text")
SPIKE_LEVELS = {"spikeminilm": 4, "spikeminilm16": 16}
SMOOTHSPIKE_CODE = os.environ.get("SW_SMOOTHSPIKE_CODE", "/home/user01/Minko/spike-new-vla-icml/code")
TEXT_DIMS = {"minilm": 384, "spikeminilm": 384, "spikeminilm16": 384, "smoothspike": 768, "bert": 768}
MAX_TOKENS = 32
CHUNK = 2048       # texts encoded per forward pass


def _path():
    return sorted(glob.glob(os.path.join(MINILM, "*")))[0]


def _smoothspike():
    """The backbone of the fused SmoothSpike-BERT checkpoint, loaded strictly (as turbovla/text/smoothspike.py)."""
    from safetensors.torch import load_file
    from transformers import BertConfig
    if SMOOTHSPIKE_CODE not in sys.path:
        sys.path.insert(0, SMOOTHSPIKE_CODE)
    from third_party.SmoothSpike.spikingbert_rot_inf import BertForMaskedLM
    cfg = BertConfig.from_pretrained(SMOOTHSPIKE, local_files_only=True)
    cfg._attn_implementation = "eager"
    model = BertForMaskedLM(cfg)
    state = load_file(os.path.join(SMOOTHSPIKE, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}, \
        (missing, unexpected)
    return model.bert


class TextEncoder:
    def __init__(self, device="cpu", pool="tokens", encoder="minilm", max_tokens=MAX_TOKENS):
        from transformers import AutoModel, AutoTokenizer
        path = {"smoothspike": SMOOTHSPIKE, "bert": "bert-base-uncased"}.get(encoder, _path())
        self.tok = AutoTokenizer.from_pretrained(path)
        if encoder in SPIKE_LEVELS:
            from .spike_text import SpikeMiniLM
            self.model = SpikeMiniLM.load(os.path.join(SPIKE_TEXT, f"spike_minilm_d{SPIKE_LEVELS[encoder]}"))
        else:
            self.model = _smoothspike() if encoder == "smoothspike" else AutoModel.from_pretrained(path)
        self.model = self.model.to(device).eval()
        self.device, self.pool, self.dim, self.max_tokens = device, pool, TEXT_DIMS[encoder], max_tokens
        self.cache = {}

    @torch.no_grad()
    def __call__(self, texts):
        """Context (B, L, dim) float32 and mask (B, L) bool; L = max_tokens (default MAX_TOKENS) for tokens, 1 for mean."""
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for s in range(0, len(missing), CHUNK):
            part = missing[s:s + CHUNK]
            # The sentence embedding reads the whole instruction (LIBERO-Plus rewrites run past 32 tokens).
            enc = self.tok(part, padding="max_length", truncation=True,
                           max_length=self.max_tokens if self.pool == "tokens" else 128, return_tensors="pt").to(self.device)
            hid = self.model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].bool()
            if self.pool == "mean":
                sent = (hid * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                hid = torch.nn.functional.normalize(sent, dim=-1)[:, None] * self.dim ** 0.5
                mask = mask[:, :1]
            for i, t in enumerate(part):
                self.cache[t] = (hid[i], mask[i])
        return (torch.stack([self.cache[t][0] for t in texts]), torch.stack([self.cache[t][1] for t in texts]))
