"""Closed-loop policy around a trained checkpoint: skeleton in, executable LIBERO actions out."""

from __future__ import annotations

import json
import os
import time
from collections import deque

import numpy as np
import torch

from .data import RT_ARM, RT_PROPRIO_NODES, Normalizer
from .text import TextEncoder
from .train import MODELS


def retry_oom(fn, *args, tries=60, wait=30, **kwargs):
    """fn(*args, **kwargs), called again after `wait` s while the GPU is out of memory (evaluations share GPUs with
    data collection, whose memory use moves)."""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:          # torch.cuda.OutOfMemoryError, CUDA "out of memory", cuBLAS ALLOC_FAILED
            if not any(m in str(e) for m in ("out of memory", "ALLOC_FAILED")) or i == tries - 1:
                raise
            torch.cuda.empty_cache()
            time.sleep(wait)


class SkeletonPolicy:
    def __init__(self, ckpt_path, device="cuda", weights="ema", k=3, h=10, n_exec=16, seed=0, consensus="medoid",
                 flow_steps=0, fixed_noise=False, temporal=0.0, grip_bin=False):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cls, cfg_cls = MODELS[ck["model_name"]]
        self.model = cls(cfg_cls(**ck["cfg"])).to(device).eval()
        self.model.load_state_dict(ck[weights])
        if flow_steps:                     # Euler steps per sample (0: the checkpoint's)
            self.model.cfg.flow_steps = flow_steps
        with open(os.path.join(os.path.dirname(ckpt_path), "stats.json")) as f:
            self.norm = Normalizer(json.load(f))
        self.text = TextEncoder(device, pool=self.model.cfg.text_pool, encoder=self.model.cfg.text_encoder)
        self.device, self.k, self.h, self.n_exec, self.consensus = device, k, h, n_exec, consensus
        self.fixed_noise, self.noise = fixed_noise, None
        self.temporal, self.tail = temporal, None
        self.grip_bin = grip_bin
        self.units = None if self.norm.rel_scale is None else torch.as_tensor(
            self.norm.units(self.model.cfg.horizon), device=device)
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def set_scene(self, extractor, instruction):
        """Node descriptors of the scene's skeleton and the instruction's tokens."""
        retry_oom(self._set_scene, extractor, instruction)

    def _set_scene(self, extractor, instruction):
        n, n_max = extractor.n_nodes, self.model.cfg.n_max
        pad = lambda x: torch.as_tensor(np.concatenate([x, np.zeros(n_max - n, np.int64)]), device=self.device)
        self.n_nodes = n
        self.desc = [pad(extractor.node_id)[None], pad(extractor.role)[None], pad(extractor.slot)[None]]
        self.node_mask = (torch.arange(n_max, device=self.device) < n)[None]
        self.text_tokens, self.text_mask = self.text([instruction])
        self.past = deque(maxlen=self.model.cfg.motion_lag + 1)
        self.tail = None
        if self.fixed_noise:               # one starting noise per episode, reused at every replanning step
            self.noise = torch.randn(self.k, self.model.cfg.horizon, self.model.cfg.action_dim, device=self.device,
                                     generator=self.gen)

    def observe(self, skeleton):
        """Record the skeleton of every control step (the motion input reads the one motion_lag steps back)."""
        if self.model.cfg.motion_lag:
            self.past.append(self.norm.pos(skeleton))

    @torch.no_grad()
    def plan(self, skeleton, joints=None):
        """The first n_exec actions of a sampled chunk for one (n_nodes, 3) skeleton in the robot frame; a
        posture-relative model (stats with rel_scale) also needs the current joint vector, the anchor of the chunk."""
        self.chunk, self.t0 = retry_oom(self._sample, skeleton, joints), 0
        acts = self.norm.act_inv(self.chunk[0].cpu().numpy(), joints)
        if self.grip_bin and self.norm.rel_scale is not None:
            # RoboTwin grippers (continuous): open or closed by the midpoint, as LIBERO decodes its gripper by sign
            g = [i for i in range(acts.shape[-1]) if i not in RT_ARM]
            lo, hi = self.norm.act_lo[g], self.norm.act_hi[g]
            acts[:, g] = np.where(acts[:, g] > (lo + hi) / 2, hi, lo)
        out = acts[:self.n_exec].copy()
        if self.temporal and self.tail is not None:
            # leaky readout: the motion dimensions keep a share `temporal` of the previous plan for the same steps
            md = self.norm.md
            out[:, :md] = (1 - self.temporal) * out[:, :md] + self.temporal * self.tail[:, :md]
        self.tail = acts[self.n_exec:2 * self.n_exec] if self.temporal else None
        return out

    def _sample(self, skeleton, joints):
        cur = self.encode(skeleton, joints)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.model.sample_actions(cur, *self.desc, self.node_mask, self.text_tokens, self.text_mask,
                                          k=self.k, h=self.h, generator=self.gen, units=self.units,
                                          consensus=self.consensus, noise=self.noise)
        a = a.float()
        torch.cuda.synchronize()           # so that a CUDA error surfaces inside retry_oom
        return a

    @torch.no_grad()
    def refine(self, skeleton, since, tau0=0.3):
        """Fast loop, `since` actions after the last full plan: the rest of the chunk (padded with its last action)
        re-noised to tau0 and integrated on the current skeleton; returns the actions left before the next plan."""
        assert self.norm.rel_scale is None, "refine: absolute chunks only"
        rest = self.chunk[:, since - self.t0:]
        rest = torch.cat([rest, rest[:, -1:].expand(-1, self.chunk.shape[1] - rest.shape[1], -1)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.model.refine_actions(self.encode(skeleton), *self.desc, self.node_mask, self.text_tokens,
                                          self.text_mask, rest, tau0=tau0, generator=self.gen)
        self.chunk, self.t0 = a.float(), since
        return self.norm.act_inv(a[0].float().cpu().numpy())[:self.n_exec - since]

    def encode(self, skeleton, joints=None):
        lag, proprio = self.model.cfg.motion_lag, self.model.cfg.proprio
        cur = np.zeros((1, self.model.cfg.n_max, (6 if lag else 3) + int(proprio)), np.float32)
        pos = self.norm.pos(skeleton)
        cur[0, :self.n_nodes, :3] = pos
        if lag:                      # as in training: the oldest recorded frame until motion_lag steps have passed
            cur[0, :self.n_nodes, 3:6] = pos - (self.past[0] if self.past else pos)
        if proprio:                  # measured joint angles on the arm link nodes (data.proprio_channel)
            cur[0, RT_PROPRIO_NODES, -1] = self.norm.joints(np.asarray(joints, np.float32))
        return torch.as_tensor(cur, device=self.device)
