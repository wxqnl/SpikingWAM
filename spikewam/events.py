"""Event accounting for replanning steps, in three ways of running the same trained network (research plan 4.8):

  dense   every flow step and every MAC sample recomputes every token (what sample_actions does);
  cache   a token is recomputed only when its input changed since the previous call: the current-skeleton tokens are
          computed once per replanning step, as any implementation (ANN included) would;
  delta   change-driven firing (sigma-delta): every spiking neuron keeps its last integer output and sends only the
          change, every receiver keeps its accumulated input. Linear layers are linear, so the outputs are exactly
          those of dense; an event of change c costs |c| unit spikes.

The action stream starts afresh at every MAC sample (new noise): its neurons and accumulators reset, so its first
flow step costs as much as in dense. The current-skeleton stream carries over between flow steps, samples and
replanning steps, so it only pays for what moved. With persist=True (one sample per replanning step from the same
starting noise all episode), the action stream also carries over between replanning steps.

Counted per call: every nn.Linear (spike inputs: |s| accumulates per output, SOP; real inputs: one MAC per weight),
the attention products of the spiking model (Q K^T and A V; spike operands accumulate, real text values multiply),
and every LayerNorm with the adaLN modulation and gating around it (NORM_OPS MAC-equivalents per channel and token).
Not counted: the noise-time MLPs and adaLN tables (they depend only on the fixed flow times, so they are a lookup
table), the text keys and values (once per episode: text_once_mac_per_episode), neuron threshold comparisons.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn

MODES = ("dense", "cache", "delta")
NORM_OPS = 8        # mean, variance, normalise, adaLN scale and shift, residual gate (per channel and token)
TABLE = ("t_emb", "t_mod", "final_mod")


def is_spike(x):
    return bool(torch.all(x == torch.round(x)).item()) and x.abs().max().item() <= 32


def stream_of(name):
    if name.startswith(("cur_in", "fut_in")) or ".experts.0." in name or name.startswith("head_spike.0"):
        return "world"
    return "action"


class EventCounter:
    """Attach to a SkeleWAM / SpikeWAM; call new_episode, new_sample around sample_actions(k=1) calls."""

    def __init__(self, model, persist=False):
        self.model, self.persist = model, persist
        self.totals = {m: defaultdict(float) for m in MODES}
        self.text_once = 0.0
        self.n_episodes = 0
        self.prev = {}
        self.n_world = 0
        self.hooks = []
        self.attn_calls = defaultdict(int)
        for name, mod in model.named_modules():
            if name.split(".")[0] in TABLE:
                continue
            if isinstance(mod, nn.Linear):
                self.hooks.append(mod.register_forward_pre_hook(
                    lambda m, inp, name=name: self.linear(name, m, inp[0])))
            elif isinstance(mod, nn.LayerNorm):
                self.hooks.append(mod.register_forward_pre_hook(
                    lambda m, inp, name=name: self.norm(name, inp[0])))
        for li, layer in enumerate(model.layers):
            if hasattr(layer, "attend"):
                orig = layer.attend

                def attend(q, k, v, bias, li=li, orig=orig, **kw):
                    call = self.attn_calls[li] % 3          # self-attention, then world and action cross-attention
                    self.attn_calls[li] += 1
                    self.attention(f"L{li}.{('self', 'cross0', 'cross1')[call]}", q, k, v, bias)
                    return orig(q, k, v, bias, **kw)
                layer.attend = attend
            else:
                self.hooks.append(layer.register_forward_pre_hook(
                    lambda m, args, li=li: self.ann_attention(f"ann.L{li}", args)))

    # ---- bookkeeping ------------------------------------------------------------------------------------
    def new_episode(self, n_world):
        self.prev, self.n_world = {}, n_world
        self.n_episodes += 1

    def new_sample(self):
        """Fresh noise: the action stream's neurons and accumulators reset."""
        if self.persist:
            return
        for key in list(self.prev):
            if key.endswith((".cross0", "ckv")) or key.startswith("ann.") or key == "final_norm.calls":
                continue
            if key.endswith(".cross1") or (not key.endswith(".self") and stream_of(key) == "action"):
                del self.prev[key]
            elif key.endswith(".self"):
                p = self.prev[key]
                for t in ("q", "k", "v"):
                    p[t][:, :, self.n_world:] = 0
                p["A"][:, :, self.n_world:] = float("nan")

    def add(self, stream, kind, dense, cache, delta):
        for mode, val in zip(MODES, (dense, cache, delta)):
            self.totals[mode][f"{stream}_{kind}"] += float(val)

    def result(self, n_replans, steps_per_replan, e_mac=4.6e-12, e_ac=0.9e-12):
        out = {}
        for mode in MODES:
            t = {k: v / n_replans for k, v in self.totals[mode].items()}
            sop = sum(v for k, v in t.items() if k.endswith("_sop"))
            mac = sum(v for k, v in t.items() if k.endswith("_mac"))
            uj = (sop * e_ac + mac * e_mac) * 1e6
            out[mode] = {"per_replan": t, "sop": sop, "mac": mac, "energy_uJ_per_replan": uj,
                         "energy_uJ_per_control_step": uj / steps_per_replan}
        out["text_once_mac_per_episode"] = self.text_once / max(self.n_episodes, 1)
        return out

    # ---- linear layers and norms ------------------------------------------------------------------------
    def linear(self, name, mod, x):
        x = x.detach().float()
        if name.endswith("ckv"):
            if name not in self.prev:
                self.text_once += x[0].numel() * mod.out_features
                self.prev[name] = True
            return
        prev = self.prev.get(name)
        if prev is None or prev.shape != x.shape:
            prev = torch.zeros_like(x)
        spike = is_spike(x)
        changed = (x != prev).any(-1)
        if spike:
            full, diff = x.abs().sum(-1), (x - prev).abs().sum(-1)
        else:
            full, diff = torch.full(x.shape[:-1], x.shape[-1], device=x.device), (x != prev).sum(-1).float()
        n = mod.out_features
        self.add(stream_of(name), "sop" if spike else "mac",
                 full.sum() * n, (full * changed).sum() * n, diff.sum() * n)
        self.prev[name] = x

    def norm(self, name, x):
        if name == "final_norm":
            # Called for the world tokens, then the action tokens; the world tokens' output is unused at inference.
            calls = self.prev.get("final_norm.calls", 0)
            self.prev["final_norm.calls"] = calls + 1
            if calls % 2 == 0:
                return
        x = x.detach().float()
        prev = self.prev.get(name)
        changed = (x != prev).any(-1) if prev is not None and prev.shape == x.shape else torch.ones(
            x.shape[:-1], dtype=torch.bool, device=x.device)
        d = x.shape[-1] * NORM_OPS
        self.add(stream_of(name), "mac", x[..., 0].numel() * d, changed.sum() * d, changed.sum() * d)
        self.prev[name] = x

    # ---- attention --------------------------------------------------------------------------------------
    def attention(self, key, q, k, v, bias):
        from .spike_wam import shiftmax
        q, k, v = q.detach().float(), k.detach().float(), v.detach().float()
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + bias
        a = shiftmax(scores, self.model.cfg.attn_bits).detach()
        valid = torch.isfinite(scores.expand(q.shape[0], q.shape[1], -1, -1))
        p = self.prev.get(key)
        if p is None or p["q"].shape != q.shape or p["k"].shape != k.shape:
            p = {"q": torch.zeros_like(q), "k": torch.zeros_like(k), "v": torch.zeros_like(v),
                 "A": torch.full_like(a, float("nan"))}
        k_spike, v_spike = is_spike(k), is_spike(v)
        dh = q.shape[-1]
        dq = (q - p["q"]).abs().sum(-1)                                   # (B, H, Lq)
        dk = (k - p["k"]).abs().sum(-1) if k_spike else (k != p["k"]).any(-1).float() * dh
        dv = (v - p["v"]).abs().sum(-1) if v_spike else (v != p["v"]).any(-1).float() * dh
        key_changed = (k != p["k"]).any(-1) | (v != p["v"]).any(-1)       # (B, H, Lk)
        a_changed = ~((a == p["A"]) | ~valid).all(-1)                     # (B, H, Lq)
        q_changed = (q != p["q"]).any(-1)
        qa, va = q.abs().sum(-1), (v.abs().sum(-1) if v_spike else torch.full(v.shape[:-1], dh, device=v.device))
        if key.endswith(".self"):
            groups = (("world", slice(0, self.n_world)), ("action", slice(self.n_world, None)))
        else:
            groups = (("world" if key.endswith("cross0") else "action", slice(None)),)
        for stream, sl in groups:
            vld = valid[:, :, sl].float()
            nk, nq = vld.sum(-1), vld.sum(-2)                             # (B, H, Lq'), (B, H, Lk)
            row_changed = q_changed[:, :, sl] | ((vld > 0) & key_changed[:, :, None, :]).any(-1)
            av_row = (vld * va[:, :, None, :]).sum(-1)                    # cost of recomputing one output row
            ac = a_changed[:, :, sl]
            nq_same = (vld * (~ac)[..., None].float()).sum(-2)
            # Q K^T: a unit query spike adds one key entry per valid key (real keys: one multiply-accumulate).
            qk = [(qa[:, :, sl] * nk).sum(), (qa[:, :, sl] * nk * row_changed).sum(),
                  (dq[:, :, sl] * nk).sum() + (dk * nq).sum()]
            self.add(stream, "sop", *qk)
            # A V: a unit value spike adds one weight per valid query; real values: one MAC per weight and channel.
            av = [(av_row).sum(), (av_row * row_changed).sum(), (av_row * ac).sum() + (dv * nq_same).sum()]
            self.add(stream, "sop" if v_spike else "mac", *av)
        self.prev[key] = {"q": q, "k": k, "v": v, "A": a}

    def ann_attention(self, key, args):
        """ANN attention (scaled_dot_product_attention inside MoTLayer): 2 d MACs per valid query-key pair, per
        token-level change for cache; no delta mode for real-valued activations."""
        xs, _, mask, text, text_mask = args
        d = xs[0].shape[-1]
        n_w = xs[0].shape[1]
        valid = mask.expand(-1, 1, -1, -1)[:, 0].float() if mask.dtype == torch.bool else torch.isfinite(mask).float()
        world_new = self.prev.get(key) is None or not torch.equal(self.prev[key], xs[0])
        pw, pa = valid[:, :n_w].sum() * 2 * d, valid[:, n_w:].sum() * 2 * d
        lt = text_mask.sum(-1).float()                                     # (B,)
        cw, ca = (lt * n_w).sum() * 2 * d, (lt * xs[1].shape[1]).sum() * 2 * d
        self.add("world", "mac", pw + cw, (pw + cw) * world_new, (pw + cw) * world_new)
        self.add("action", "mac", pa + ca, pa + ca, pa + ca)
        self.prev[key] = xs[0].detach().clone()
