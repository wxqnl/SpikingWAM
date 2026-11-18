"""SpikeWAM: the skeleton world-action model with spikes between all layers.

Same tokens, masks, losses and sampling as SkeleWAM (spikewam.wam); what changes:
  * every weight layer reads integer spikes (IntSpike, D levels = D timesteps at inference): the residual stream
    carries membrane current and each block fires before its weights (membrane shortcut, as in Spike-driven
    Transformer V2/V3); Q, K and V are spikes; the output heads read spikes;
  * attention weights come from Shiftmax (as in Spike-VLA, after Bipolar Self-attention, NeurIPS 2025): scores are
    rounded to 1/2**bits and normalised by powers of two, so the weights are a small table times a shift;
  * competitive selection (`wta_heads` > 0, research plan 4.10): in the first wta_heads heads of the self-attention,
    the keys race and the k with the largest membrane score win (keys tied with the k-th win too) and share the
    read-out equally, as under lateral inhibition; the rest are silenced. Straight-through: the backward pass is
    that of Shiftmax. With `wta_pointer`, a winning skeleton node also passes its grip-centred position (3 numbers
    per head) next to its value spikes;
  * the input adapters fire after their first linear layer (direct coding of coordinates, noisy actions and noisy
    future frames), unless `code` gives the current skeleton as spikes in the first place;
  * with `fine_in`, the named input adapters fire with in_levels levels at a proportionally finer threshold (same
    range): a single layer of D-level neurons with one shared threshold keeps the coordinates to about 2 cm and the
    noisy action to 3-5 % of its spread, 10-25x worse than the ANN's adapters (implementation record 4.8);
  * with `rect_hidden`, the hidden neurons of every MLP (blocks and input adapters) fire for positive currents only.
    Signed spikes there leave the MLP nearly linear (round and clamp are the identity inside [-D, D]); one-sided
    firing restores the rectification that the ANN's GELU provides.
  * with `stoch_round`, every neuron is trained with unbiased stochastic rounding, and at inference (`carry`) the
    action branch keeps the remainder it did not fire from one flow step to the next instead of resetting (sigma-delta
    over the denoising steps); the skeleton branch rounds as before (implementation record 4.11).
Kept continuous: the noise-time embedding (one small MLP per flow step) and its adaLN modulation of the currents,
LayerNorm of the currents, and the frozen text features (keys and values of the cross-attention, once per episode).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spiking import IntSpike
from .wam import HAND_SCALE, ExpertLayer, MoTLayer, SkeleWAM, WAMConfig, modulate, wta


@dataclass
class SpikeWAMConfig(WAMConfig):
    levels: int = 4            # D: spike levels of every neuron (timesteps at inference)
    bipolar: bool = True       # signed spikes (-D..D), as Spike-VLA's bipolar / integer PLIF
    fine_in: str = ""          # input adapters ("cur", "act", "fut") whose hidden spikes get in_levels levels at a
    in_levels: int = 16        # proportionally finer threshold (same range): the precision of the continuous inputs
    rect_hidden: bool = False  # MLP hidden neurons (blocks and input adapters) fire for positive currents only (0..D):
                               # the rectification the ANN gets from GELU; bipolar hidden neurons leave the MLP linear
    attn_bits: int = 2         # Shiftmax exponent resolution 1/2**bits
    wta_pointer: bool = False  # winners also pass their grip-centred position
    real_neurons: str = ""     # localisation: comma list of SpikeExpertLayer.NEURONS kept continuous (identity)
    soft_attn: bool = False    # localisation: softmax instead of Shiftmax in the non-WTA heads
    stoch_round: bool = False  # train every neuron with unbiased stochastic rounding (IntSpike.stochastic; 4.11)
    carry: bool = True         # with stoch_round: at inference the action branch keeps its remainder across flow steps


class _Shiftmax(torch.autograd.Function):
    """Power-of-two normaliser of quantised scores; backward is the base-2 softmax Jacobian (Spike-VLA's)."""

    @staticmethod
    def forward(ctx, scores, bits):
        live = torch.isfinite(scores).any(-1, keepdim=True)
        maximum = torch.where(live, scores.amax(-1, keepdim=True), torch.zeros_like(scores[..., :1]))
        num = torch.exp2(scores - maximum)
        step = float(2 ** bits)
        den = (torch.log2(num.sum(-1, keepdim=True).clamp_min(1)) * step).ceil() / step
        w = num / torch.exp2(den)
        ctx.save_for_backward(w)
        return w

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        p = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        return math.log(2.0) * w * (g - (g * p).sum(-1, keepdim=True)), None


def shiftmax(scores, bits):
    """Scores rounded to 1/2**bits (straight through), masked entries (-inf) kept out of the rounding."""
    step = float(2 ** bits)
    live = torch.isfinite(scores)
    q = torch.where(live, scores, torch.zeros_like(scores)) * step
    q = q + (torch.round(q) - q).detach()
    return _Shiftmax.apply((q / step).masked_fill(~live, float("-inf")), bits)


class SpikeMlp(nn.Module):
    def __init__(self, dim, hidden, out, cfg):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, out)
        self.spike = IntSpike(cfg.levels, cfg.bipolar and not cfg.rect_hidden)

    def forward(self, x):
        return self.fc2(self.spike(self.fc1(x)))


class SpikeExpertLayer(ExpertLayer):
    NEURONS = ("in1", "q", "k", "v", "o", "in2", "cq", "co", "in3", "hid")

    def __init__(self, cfg):
        super().__init__(cfg)
        real = {r for r in cfg.real_neurons.split(",") if r}
        assert real <= set(self.NEURONS), f"unknown neurons {real - set(self.NEURONS)}"
        self.n = nn.ModuleDict({k: nn.Identity() if k in real else IntSpike(
            cfg.levels, cfg.bipolar and not (k == "hid" and cfg.rect_hidden)) for k in self.NEURONS})


class SpikeMoTLayer(MoTLayer):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.heads, self.bits, self.soft = cfg.heads, cfg.attn_bits, cfg.soft_attn
        self.wta_heads, self.wta_k = cfg.wta_heads, cfg.wta_k
        self.experts = nn.ModuleList([SpikeExpertLayer(cfg), SpikeExpertLayer(cfg)])
        self.ptr = nn.ModuleList([nn.Linear(3 * cfg.wta_heads, cfg.dim) for _ in range(2)]) if (
            cfg.wta_pointer and cfg.wta_heads) else None
        self.key_pos = None          # (B, L, 3) grip-centred key positions, set by SpikeWAM.attention_masks
        self.weights = None          # attention weights of the last self-attention call

    def attend(self, q, k, v, bias, select=False):
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias
        a = scores.float().softmax(-1) if self.soft else shiftmax(scores.float(), self.bits)
        if select and self.wta_heads:
            h = self.wta_heads
            with torch.no_grad():   # the race in float32, so that ties are exact
                exact = (q[:, :h].float() @ k[:, :h].float().transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias[:, :h]
            hard = wta(exact, self.wta_k)
            a = torch.cat([a[:, :h] + (hard - a[:, :h]).detach(), a[:, h:]], 1)
            self.weights = a
        return a.to(v.dtype) @ v

    def forward(self, xs, mods, mask, text, text_mask):
        m = [mod + e.mod_table for mod, e in zip(mods, self.experts)]
        q, k, v = [], [], []
        for x, e, mm in zip(xs, self.experts, m):
            s = e.n["in1"](modulate(e.norm1(x), mm[:, :, 0], mm[:, :, 1]))
            qq, kk, vv = e.qkv(s).chunk(3, -1)
            q.append(self.split(e.n["q"](qq)))
            k.append(self.split(e.n["k"](kk)))
            v.append(self.split(e.n["v"](vv)))
        bias = mask if mask.dtype != torch.bool else torch.zeros(mask.shape, device=mask.device).masked_fill(
            ~mask, float("-inf"))
        o = self.attend(torch.cat(q, 2), torch.cat(k, 2), torch.cat(v, 2), bias, select=True)
        sizes = [x.shape[1] for x in xs]
        o = o.transpose(1, 2).flatten(2).split(sizes, 1)
        ptr = [None, None]
        if self.ptr is not None:
            p = self.weights[:, :self.wta_heads] @ self.key_pos[:, None].float()     # (B, h, L, 3)
            ptr = p.transpose(1, 2).flatten(2).to(o[0].dtype).split(sizes, 1)
        text_bias = torch.zeros(text_mask.shape, device=text.device).masked_fill(~text_mask, float("-inf"))
        out = []
        for i, (x, oo, e, mm) in enumerate(zip(xs, o, self.experts, m)):
            y = e.proj(e.n["o"](oo))
            if ptr[i] is not None:
                y = y + self.ptr[i](ptr[i])
            x = x + mm[:, :, 2] * y
            s = e.n["in2"](modulate(e.norm2(x), mm[:, :, 3], mm[:, :, 4]))
            ck, cv = e.ckv(text).chunk(2, -1)
            co = self.attend(self.split(e.n["cq"](e.cq(s))), self.split(ck), self.split(cv),
                             text_bias[:, None, None, :])
            x = x + mm[:, :, 5] * e.cproj(e.n["co"](co.transpose(1, 2).flatten(2)))
            s = e.n["in3"](modulate(e.norm3(x), mm[:, :, 6], mm[:, :, 7]))
            x = x + mm[:, :, 8] * e.mlp.fc2(e.n["hid"](e.mlp.fc1(s)))
            out.append(x)
        return out


class SpikeWAM(SkeleWAM):
    def __init__(self, cfg: SpikeWAMConfig):
        super().__init__(cfg)
        d = cfg.dim
        self.cur_in = SpikeMlp(self.cur_in.fc1.in_features, d, d, cfg)
        self.fut_in = SpikeMlp(self.fut_in.fc1.in_features, d, d, cfg)
        self.act_in = SpikeMlp(self.act_in.fc1.in_features, d, d, cfg)
        self.layers = nn.ModuleList([SpikeMoTLayer(cfg) for _ in range(cfg.depth)])
        self.head_spike = nn.ModuleList([IntSpike(cfg.levels, cfg.bipolar), IntSpike(cfg.levels, cfg.bipolar)])
        fine = {f for f in cfg.fine_in.split(",") if f}
        assert fine <= {"cur", "act", "fut"}, f"unknown adapters {fine}"
        for name in fine:
            getattr(self, name + "_in").spike = IntSpike(cfg.in_levels, cfg.bipolar and not cfg.rect_hidden,
                                                         threshold=0.5 * cfg.levels / cfg.in_levels)
        self.init_weights()
        for neuron in self.spike_neurons():
            neuron.stochastic = cfg.stoch_round
        for layer in self.layers:      # the pointer starts silent
            for lin in layer.ptr or []:
                nn.init.zeros_(lin.weight)

    def attention_masks(self, cur, node_mask, fut_valid, n_fut, n_act):
        masks = super().attention_masks(cur, node_mask, fut_valid, n_fut, n_act)
        if self.cfg.wta_pointer and self.cfg.wta_heads:
            pos = self.metric(cur)
            g = self.grips[0]
            rel = (pos - pos[:, g:g + 1]) / HAND_SCALE
            key_pos = F.pad(rel, (0, 0, 0, n_fut + n_act))                     # future and action keys: 0
            for layer in self.layers:
                layer.key_pos = key_pos
        return masks

    def head_input(self, expert, h):
        return self.head_spike[expert](h)

    def spike_neurons(self):
        return [mod for mod in self.modules() if isinstance(mod, IntSpike)]

    def action_neurons(self):
        """The neurons that fire anew at every flow step: noisy-action adapter, action expert, action head (the
        skeleton branch sees the same input at every step and is computed once per plan)."""
        experts = [layer.experts[1].n[k] for layer in self.layers for k in SpikeExpertLayer.NEURONS]
        return [m for m in [self.act_in.spike, *experts, self.head_spike[1]] if isinstance(m, IntSpike)]

    @contextmanager
    def carrying(self):
        """With stoch_round and carry: the action branch keeps its remainder across the calls inside (the flow steps
        of one plan), starting from zero."""
        neurons = self.action_neurons() if self.cfg.stoch_round and self.cfg.carry else []
        for m in neurons:
            m.carry = 0
        try:
            yield
        finally:
            for m in neurons:
                m.carry = None

    @torch.no_grad()
    def sample_actions(self, *args, **kwargs):
        with self.carrying():
            return super().sample_actions(*args, **kwargs)
