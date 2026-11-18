"""SkeleWAM (arXiv 2610.02120), our re-implementation from the paper.

Two-expert Mixture-of-Transformers as in Fast-WAM: every layer holds a world expert (current skeleton tokens and
future skeleton tokens, shared weights) and an action expert (noisy action tokens). Each expert has its own
noise-time modulation, attention projections, cross-attention to the language tokens and FFN; the self-attention
runs jointly over all tokens under the block mask of Eq. (block_attention_mask):

            cur  act  fut
    cur  [   1    0    0 ]
    act  [   1    1    0 ]
    fut  [   1    0    1 ]

so the future branch can be dropped at inference without changing the action output. Both branches are trained with
flow matching (noise time 0 = data, 1 = Gaussian noise); inference integrates the action field from 1 to 0 and picks
one of K samples by Medoid Action Consensus.

Paper settings: H_a = 32, H_s = 8, 10 integration steps, H_e = 16, K = 3, h = 10, 60K steps, batch 48.
Not given in the paper (our choices): width 512, 6 layers, 8 heads, FFN ratio 3.25, one token per skeleton node for
the current skeleton and one token per future frame (as in the framework figure), adaLN-single time modulation,
future offsets 4, 8, ..., 32 steps, lambda_s = 1, noise times from the pi0 Beta(1.5, 1) schedule.

Geometry options (shared with the spiking model, so that the ANN + geometry control differs from R0 only there):
`code` replaces the coordinate MLP input of the current skeleton by a spike code of the metric positions
(spikewam.spiking.CODES); `geo_attn` adds the coincidence-attention bias (spikewam.geometry); `frame` gives each
node its position relative to the grip site ("hand", in units of HAND_SCALE for direct input, metres for a code)
instead of, or next to ("both"), its position in the robot frame.
`motion_lag` gives every node its displacement over the last motion_lag control steps as a second input (in units
of MOTION_UNIT), next to its position: the current skeleton is then (B, N, 6), positions then displacements, both in
normalised units.
Bimanual scenes (RoboTwin) have one grip site per arm (`grip_nodes`): "hand" gives the position relative to each of
them, and the coincidence attention of action and future-frame tokens anchors the first half of the heads at the
first grip site and the second half at the second. The benchmark fields (action size, node slots, embedding tables)
default to LIBERO.
Future-branch options (research plan E4): `future_tokens="node"` gives every node of every future frame its own token
(node identity embeddings, anchored at the node's current position); `future_target="code"` replaces flow matching on
future coordinates by classifying, on every phase ring of the grid code, the phase bin of each node's future
displacement (the future tokens are then queries without noisy input).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ACTION_DIM, MOTION_DIMS, N_MAX
from .geometry import GRIP_NODE, CoincidenceGeometry
from .skeleton import MAX_SLOTS, N_NODE_IDS, N_ROLES
from .spiking import CODES, GridSpikeCode
from .text import TEXT_DIMS

N_MOD = 9  # shift, scale, gate for self-attention, cross-attention and FFN
HAND_SCALE = 0.2  # metres; scale of hand-centred direct coordinates
MOTION_UNIT = 0.01  # metres; scale of node displacements (motion_lag input)


@dataclass
class WAMConfig:
    dim: int = 512
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 3.25    # 51.4M trainable parameters, the size of SkeleWAM (sim-state)
    horizon: int = 32          # H_a
    n_future: int = 8          # H_s
    lambda_s: float = 1.0
    flow_steps: int = 10
    use_future: bool = True    # False: action-only training (paper ablation b)
    code: str = "direct"       # current-skeleton input: "direct" coordinates or a spike code of metric positions
    frame: str = "abs"         # "abs" robot frame, "hand" relative to the grip site, "both"
    geo_attn: bool = False     # coincidence-attention bias over current-skeleton keys
    geo_shift: bool = True     # learned per-head offset Delta_h of the coincidence attention
    text_pool: str = "tokens"  # language context: "tokens" (token states) or "mean" (one sentence embedding)
    text_encoder: str = "minilm"  # frozen language encoder, one of spikewam.text.TEXT_DIMS
    future_tokens: str = "frame"  # "frame": one token per future frame; "node": one per node and frame
    future_target: str = "flow"   # "flow": flow matching on coordinates; "code": grid-code phase classification
    wta_heads: int = 0         # self-attention heads with k-winner-take-all read-out (the first ones; 4.10)
    wta_k: int = 1             # winners per query in those heads
    motion_lag: int = 0        # node displacement over this many control steps as a second input (0: positions only)
    posture_rel: bool = False  # RoboTwin: arm joints of the chunk relative to the current joint vector (data.py; 4.13)
    posture_scale: str = "step"   # their scale: per chunk step and joint ("step") or per joint ("joint")
    proprio: bool = False      # RoboTwin: each arm link node also carries its measured joint angle (data.py; 4.13)
    pos_mean: tuple = (0.0, 0.0, 0.0)  # coordinate normalisation (set from the data), to recover metres
    pos_std: tuple = (1.0, 1.0, 1.0)
    # Benchmark (set from the data; defaults are LIBERO's).
    action_dim: int = ACTION_DIM
    motion_dims: int = MOTION_DIMS     # leading action dimensions compared by Medoid Action Consensus
    n_max: int = N_MAX                 # node slots per skeleton
    grip_nodes: tuple = (GRIP_NODE,)   # grip site node(s), one per arm
    n_node_ids: int = N_NODE_IDS
    n_roles: int = N_ROLES
    max_slots: int = MAX_SLOTS


def wta(scores, k):
    """k-winner-take-all weights over the valid keys: the k largest scores and every key tied with the k-th share
    the read-out equally; masked keys (-inf) never win."""
    live = torch.isfinite(scores)
    kth = scores.topk(min(k, scores.shape[-1]), -1).values[..., -1:]
    win = (scores >= kth) & live
    return win.to(scores.dtype) / win.sum(-1, keepdim=True).clamp_min(1)


def timestep_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[..., None] * 1000 * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Mlp(nn.Module):
    def __init__(self, dim, hidden, out=None):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, out or dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class ExpertLayer(nn.Module):
    """One expert's weights inside a MoT layer."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.cq, self.ckv, self.cproj = nn.Linear(d, d), nn.Linear(TEXT_DIMS[cfg.text_encoder], 2 * d), nn.Linear(d, d)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(d, int(d * cfg.mlp_ratio))
        self.mod_table = nn.Parameter(torch.randn(N_MOD, d) / d ** 0.5)


class MoTLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.wta_heads, self.wta_k = cfg.heads, cfg.wta_heads, cfg.wta_k
        self.experts = nn.ModuleList([ExpertLayer(cfg), ExpertLayer(cfg)])  # 0 world, 1 action

    def split(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.heads, -1).transpose(1, 2)

    def forward(self, xs, mods, mask, text, text_mask):
        """xs: [world tokens, action tokens]; mods: matching (B, L, N_MOD, d); mask: (B, 1, L, L) over the
        concatenation [world, action]."""
        m = [mod + e.mod_table for mod, e in zip(mods, self.experts)]
        qkv = [e.qkv(modulate(e.norm1(x), mm[:, :, 0], mm[:, :, 1])) for x, e, mm in zip(xs, self.experts, m)]
        q, k, v = (torch.cat([self.split(t.chunk(3, -1)[j]) for t in qkv], 2) for j in range(3))
        if self.wta_heads:
            o = self.select_attention(q, k, v, mask).transpose(1, 2).flatten(2)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).flatten(2)
        o = o.split([x.shape[1] for x in xs], 1)
        out = []
        for x, oo, e, mm in zip(xs, o, self.experts, m):
            x = x + mm[:, :, 2] * e.proj(oo)
            h = modulate(e.norm2(x), mm[:, :, 3], mm[:, :, 4])
            ck, cv = e.ckv(text).chunk(2, -1)
            co = F.scaled_dot_product_attention(self.split(e.cq(h)), self.split(ck), self.split(cv),
                                                attn_mask=text_mask[:, None, None, :])
            x = x + mm[:, :, 5] * e.cproj(co.transpose(1, 2).flatten(2))
            x = x + mm[:, :, 8] * e.mlp(modulate(e.norm3(x), mm[:, :, 6], mm[:, :, 7]))
            out.append(x)
        return out


    def select_attention(self, q, k, v, mask):
        """Softmax attention whose first wta_heads heads read out their k-WTA winners (straight-through: softmax
        gradient), the ANN control of competitive selection."""
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        a = scores.softmax(-1)
        h = self.wta_heads
        hard = wta(scores[:, :h].detach(), self.wta_k)
        a = torch.cat([a[:, :h] + (hard - a[:, :h]).detach(), a[:, h:]], 1)
        return a.to(v.dtype) @ v


class SkeleWAM(nn.Module):
    def __init__(self, cfg: WAMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        # Current skeleton encoder phi_c: per-node coordinate MLP (or spike code) plus node identity embeddings.
        self.code = None if cfg.code == "direct" else CODES[cfg.code]()
        self.grips = list(cfg.grip_nodes)
        n_frames = (cfg.frame in ("abs", "both")) + len(self.grips) * (cfg.frame in ("hand", "both"))
        self.cur_in = Mlp((3 if self.code is None else self.code.dim) * n_frames + 3 * (cfg.motion_lag > 0)
                          + int(cfg.proprio), d, d)
        self.register_buffer("pos_mean", torch.tensor(cfg.pos_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("pos_std", torch.tensor(cfg.pos_std, dtype=torch.float32), persistent=False)
        self.geo = CoincidenceGeometry(cfg.heads, cfg.depth, shift=cfg.geo_shift) if cfg.geo_attn else None
        self.node_emb = nn.Embedding(cfg.n_node_ids, d)
        self.role_emb = nn.Embedding(cfg.n_roles, d)
        self.slot_emb = nn.Embedding(cfg.max_slots + 1, d)
        # Future skeleton adapter phi_s: one token per future frame (or per node and frame).
        node_fut = cfg.future_tokens == "node"
        self.fut_in = Mlp(3 if node_fut else cfg.n_max * 3, d, d)
        self.fut_pos = nn.Parameter(torch.zeros(cfg.n_future, d))
        if cfg.future_target == "code":
            assert node_fut, "the code target needs node-level future tokens"
            self.fut_code = GridSpikeCode()
            fut_out_dim = self.fut_code.dim
        else:
            fut_out_dim = 3 if node_fut else cfg.n_max * 3
        # Action adapter phi_a: one token per action step.
        self.act_in = Mlp(cfg.action_dim, d, d)
        self.act_pos = nn.Parameter(torch.zeros(cfg.horizon, d))
        # Noise-time modulation (adaLN-single): one projection per expert, a learned offset table per layer.
        self.t_emb = Mlp(256, d, d)
        self.t_mod = nn.ModuleList([nn.Linear(d, N_MOD * d), nn.Linear(d, N_MOD * d)])
        self.layers = nn.ModuleList([MoTLayer(cfg) for _ in range(cfg.depth)])
        self.final_mod = nn.ModuleList([nn.Linear(d, 2 * d), nn.Linear(d, 2 * d)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.fut_out = nn.Linear(d, fut_out_dim)
        self.act_out = nn.Linear(d, cfg.action_dim)
        self.init_weights()

    def init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
        for emb in (self.node_emb, self.role_emb, self.slot_emb):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.fut_pos, std=0.02)
        nn.init.normal_(self.act_pos, std=0.02)
        for lin in list(self.t_mod) + list(self.final_mod):  # adaLN-zero: every block starts as identity
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in (self.fut_out, self.act_out):
            nn.init.zeros_(lin.weight)

    def head_input(self, expert, h):
        """What the output heads read (the spiking model fires here)."""
        return h

    # ---- token builders -------------------------------------------------------------------------------
    def metric(self, cur):
        return cur[..., :3].float() * self.pos_std + self.pos_mean

    def identity(self, node_id, role, slot):
        return self.node_emb(node_id) + self.role_emb(role) + self.slot_emb(slot)

    def encode_current(self, cur, ident):
        pos = self.metric(cur)
        parts = []
        if self.cfg.frame in ("abs", "both"):
            parts.append(cur[..., :3] if self.code is None else self.code(pos))
        if self.cfg.frame in ("hand", "both"):
            for g in self.grips:
                rel = pos - pos[:, g:g + 1]
                parts.append(rel / HAND_SCALE if self.code is None else self.code(rel))
        if self.cfg.motion_lag:
            parts.append(cur[..., 3:6].float() * self.pos_std / MOTION_UNIT)
        if self.cfg.proprio:
            parts.append(cur[..., -1:])
        return self.cur_in(torch.cat(parts, -1).to(cur.dtype)) + ident

    def future_tokens(self, fut_noisy, tau_s, ident):
        """Future tokens (B, L_f, d), their noise-time tokens and key validity (B, L_f) or None (all valid)."""
        hs, (b, n) = self.cfg.n_future, ident.shape[:2]
        if self.cfg.future_tokens == "frame":
            return self.fut_in(fut_noisy.flatten(2)) + self.fut_pos[:hs], self.time_tokens(tau_s, hs), None
        if self.cfg.future_target == "code":
            x = ident[:, None].expand(-1, hs, -1, -1)
            tau_s = torch.zeros(b, device=ident.device)
        else:
            x = self.fut_in(fut_noisy) + ident[:, None]
        return (x + self.fut_pos[:hs, None]).flatten(1, 2), self.time_tokens(tau_s, hs * n), None

    def attention_masks(self, cur, node_mask, fut_valid, n_fut, n_act):
        """Per-layer attention masks: boolean (shared) without geometry, else additive float (B, H, L, L).
        Query anchors: current nodes and node-level future tokens use the node's current position; future frames and
        actions use a grip site (head h of H: grip h * G // H of G). Overlaps are computed once per distinct anchor
        (the nodes and the grip sites)."""
        mask = self.block_mask(node_mask, fut_valid, n_fut, n_act)
        if self.geo is None:
            return [mask] * self.cfg.depth
        b, n = node_mask.shape
        heads, n_grip = self.cfg.heads, len(self.grips)
        pos = self.metric(cur)
        ov = self.geo.overlaps(torch.cat([pos, pos[:, self.grips]], 1), pos)   # (B, H, N + G, N, R)
        node_rows = torch.arange(n, device=cur.device)
        grip_row = n + torch.arange(heads, device=cur.device)[:, None] * n_grip // heads       # (H, 1)
        fut_rows = node_rows.repeat(n_fut // n).expand(heads, -1) if self.cfg.future_tokens == "node" else \
            grip_row.expand(-1, n_fut)
        rows = torch.cat([node_rows.expand(heads, -1), fut_rows, grip_row.expand(-1, n_act)], 1)   # (H, L)
        hh = torch.arange(heads, device=cur.device)[:, None]
        base = torch.zeros(mask.shape, device=cur.device).masked_fill(~mask, float("-inf"))
        out = []
        for layer in range(self.cfg.depth):
            g = self.geo.bias(ov, layer)[:, hh, rows]                # (B, H, L, N)
            out.append(base + torch.nn.functional.pad(g, (0, n_fut + n_act)))
        return out

    def time_tokens(self, tau, n):
        return self.t_emb(timestep_embedding(tau))[:, None].expand(-1, n, -1)

    def forward(self, cur, node_id, role, slot, node_mask, text, text_mask, act_noisy, tau_a,
                fut_noisy=None, tau_s=None, with_future=False):
        """Vector fields for the noisy actions (B, H_a, 7) and, with the future branch (given noisy futures, or
        with_future for the code target), the future output: vector fields (B, H_s, N_MAX, 3) or phase logits
        (B, H_s, N_MAX, rings, phases)."""
        b, n = cur.shape[:2]
        ident = self.identity(node_id, role, slot)
        x_cur = self.encode_current(cur, ident)
        x_act = self.act_in(act_noisy) + self.act_pos
        world, t_world = [x_cur], [self.time_tokens(torch.zeros(b, device=cur.device), n)]
        n_fut, fut_valid = 0, None
        if fut_noisy is not None or with_future:
            x_fut, t_fut, _ = self.future_tokens(fut_noisy, tau_s, ident)
            n_fut = x_fut.shape[1]
            if self.cfg.future_tokens == "node":
                fut_valid = node_mask.repeat(1, self.cfg.n_future)
            world.append(x_fut)
            t_world.append(t_fut)
        xs = [torch.cat(world, 1), x_act]
        ts = [torch.cat(t_world, 1), self.time_tokens(tau_a, x_act.shape[1])]
        mods = [self.t_mod[e](F.silu(t)).unflatten(-1, (N_MOD, -1)) for e, t in enumerate(ts)]
        masks = self.attention_masks(cur, node_mask, fut_valid, n_fut, x_act.shape[1])
        for layer, mask in zip(self.layers, masks):
            xs = layer(xs, mods, mask, text, text_mask)
        outs = []
        for e, (x, t) in enumerate(zip(xs, ts)):
            shift, scale = self.final_mod[e](F.silu(t)).chunk(2, -1)
            outs.append(self.head_input(e, modulate(self.final_norm(x), shift, scale)))
        v_act = self.act_out(outs[1])
        v_fut = None
        if n_fut:
            f = self.fut_out(outs[0][:, n:])
            if self.cfg.future_tokens == "frame":
                v_fut = f.unflatten(-1, (self.cfg.n_max, 3))
            elif self.cfg.future_target == "code":
                v_fut = f.unflatten(1, (self.cfg.n_future, n)).unflatten(-1, (-1, self.fut_code.n_phase))
            else:
                v_fut = f.unflatten(1, (self.cfg.n_future, n))
        return v_act, v_fut

    @staticmethod
    def block_mask(node_mask, fut_valid, n_fut, n_act):
        """(B, 1, L, L) boolean mask over [current nodes, future tokens, actions]."""
        b, n = node_mask.shape
        L = n + n_fut + n_act
        dev = node_mask.device
        group = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n_fut, dtype=torch.long),
                           torch.full((n_act,), 2, dtype=torch.long)]).to(dev)
        same = group[:, None] == group[None, :]
        to_cur = (group[None, :] == 0).expand(L, L)
        allowed = (same | to_cur)[None].expand(b, L, L).clone()
        if fut_valid is None:
            fut_valid = torch.ones(b, n_fut, dtype=torch.bool, device=dev)
        key_ok = torch.cat([node_mask, fut_valid, torch.ones(b, n_act, dtype=torch.bool, device=dev)], 1)
        return (allowed & key_ok[:, None, :])[:, None]

    def future_code_target(self, cur, future):
        """Phase bin (B, H_s, N, rings) of each node's future displacement on every ring of the grid code."""
        code = self.fut_code
        disp = self.metric(future) - self.metric(cur)[:, None]
        phase = (disp @ code.axes.T)[..., :, None] / code.scales                  # (B, H_s, N, D, M)
        return torch.remainder(torch.round(phase * code.n_phase), code.n_phase).long().flatten(-2)

    # ---- training -------------------------------------------------------------------------------------
    @staticmethod
    def sample_tau(b, device):
        """pi0 schedule in our convention (1 = noise): emphasises noisy times."""
        u = torch.distributions.Beta(1.5, 1.0).sample((b,)).to(device)
        return 0.001 + 0.999 * u

    def loss(self, batch, text, text_mask, teacher=None, kd=0.0):
        """Flow-matching losses; with a teacher, also the distance of the action vector field to the teacher's on
        the same noisy actions and times (weight kd)."""
        acts, cur = batch["actions"], batch["cur"]
        b = acts.shape[0]
        mask = batch["node_mask"]
        tau_a = self.sample_tau(b, acts.device)
        eps_a = torch.randn_like(acts)
        x_a = (1 - tau_a[:, None, None]) * acts + tau_a[:, None, None] * eps_a
        fut_noisy = tau_s = eps_s = None
        code_target = self.cfg.use_future and self.cfg.future_target == "code"
        if self.cfg.use_future and not code_target:
            fut = batch["future"]
            tau_s = self.sample_tau(b, acts.device)
            eps_s = torch.randn_like(fut) * mask[:, None, :, None]
            fut_noisy = (1 - tau_s[:, None, None, None]) * fut + tau_s[:, None, None, None] * eps_s
        v_a, v_s = self(cur, batch["node_id"], batch["role"], batch["slot"], mask, text, text_mask,
                        x_a, tau_a, fut_noisy, tau_s, with_future=code_target)
        loss_a = F.mse_loss(v_a.float(), eps_a - acts)
        out = {"loss_action": loss_a, "loss": loss_a}
        if code_target:
            target = self.future_code_target(cur, batch["future"])                # (B, H_s, N, R)
            ce = F.cross_entropy(v_s.float().flatten(0, -2), target.flatten(), reduction="none").view(target.shape)
            w = mask[:, None, :, None].float().expand_as(ce)
            loss_s = (ce * w).sum() / w.sum()
        elif self.cfg.use_future:
            w = mask[:, None, :, None].float().expand_as(v_s)
            loss_s = (((v_s.float() - (eps_s - batch["future"])) ** 2) * w).sum() / w.sum()
        if self.cfg.use_future:
            out["loss_skeleton"] = loss_s
            out["loss"] = loss_a + self.cfg.lambda_s * loss_s
        if teacher is not None:
            with torch.no_grad():
                v_t, _ = teacher(cur, batch["node_id"], batch["role"], batch["slot"], mask, text, text_mask, x_a, tau_a)
            out["loss_kd"] = F.mse_loss(v_a.float(), v_t.float())
            out["loss"] = out["loss"] + kd * out["loss_kd"]
        return out

    # ---- inference ------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, cur, node_id, role, slot, node_mask, text, text_mask, k=3, h=10, generator=None,
                       units=None, consensus="medoid", noise=None):
        """Action chunks (B, H_a, 7) in normalised space; Medoid Action Consensus over k samples (distances in the
        action units per normalised unit `units` (H_a, D) if given, e.g. radians for posture-relative chunks, whose
        scale differs between chunk steps), or with consensus="mean" the average of the k samples (the gripper
        dimension, decoded by its sign, then takes the majority). `noise` (B * k, H_a, 7), if given, replaces the drawn
        starting noise."""
        b = cur.shape[0]
        rep = lambda x: x.repeat_interleave(k, 0)
        args = [rep(x) for x in (cur, node_id, role, slot, node_mask, text, text_mask)]
        x = torch.randn(b * k, self.cfg.horizon, self.cfg.action_dim, device=cur.device, generator=generator) \
            if noise is None else noise.float().clone()
        steps = self.cfg.flow_steps
        for i in range(steps):
            tau = torch.full((b * k,), 1 - i / steps, device=cur.device)
            v, _ = self(*args, x, tau)
            x = x - v.float() / steps
        x = x.view(b, k, self.cfg.horizon, self.cfg.action_dim)
        if k == 1:
            return x[:, 0]
        if consensus == "mean":
            return x.mean(1)
        md = self.cfg.motion_dims
        pre = (x[:, :, :h, :md] * (1 if units is None else units[:h, :md])).flatten(2)
        dist = torch.cdist(pre, pre) ** 2 / (h * md)
        best = (dist.sum(-1) / (k - 1)).argmin(-1)
        return x[torch.arange(b), best]

    @torch.no_grad()
    def refine_actions(self, cur, node_id, role, slot, node_mask, text, text_mask, chunk, tau0=0.3, generator=None):
        """Fast loop: noise a planned chunk (B, H_a, 7) back to noise time tau0 and integrate the last flow steps
        on the current skeleton (SDEdit-style; the model is trained on every noise time)."""
        steps = self.cfg.flow_steps
        n = max(1, round(tau0 * steps))
        eps = torch.randn(chunk.shape, device=chunk.device, generator=generator)
        x = (1 - n / steps) * chunk.float() + n / steps * eps
        for i in range(steps - n, steps):
            tau = torch.full((cur.shape[0],), 1 - i / steps, device=cur.device)
            v, _ = self(cur, node_id, role, slot, node_mask, text, text_mask, x, tau)
            x = x - v.float() / steps
        return x
