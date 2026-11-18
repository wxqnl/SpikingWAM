"""The spiking front end in RoboTwin's closed loop (RoboTwin python): object nodes from the head and two wrist cameras, arm nodes from the joints."""

from __future__ import annotations

import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..perception import PerceptConfig, SpikePercept
from .layout import FINGERS, GRIP_NODES, N_ROBOT, ROLES, SLOT_SIZE
from .perception_cache import views


class RTFrontEnd:
    def __init__(self, ckpt, device):
        # ckpt: one checkpoint, or a comma list (ensemble: the fused node estimates of all front ends are averaged before
        # the refit; they must share the object kinds; the first one's settings drive the inputs)
        paths = ckpt.split(",")
        st = torch.load(paths[0], map_location="cpu", weights_only=False)
        self.cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
        self.model = SpikePercept(self.cfg).to(device).eval()
        self.model.load_state_dict(st["model"])
        self.models = [(self.model, self.cfg)]
        for path in paths[1:]:
            s2 = torch.load(path, map_location="cpu", weights_only=False)
            c2 = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in s2["cfg"].items()})
            assert tuple(c2.kinds) == tuple(self.cfg.kinds) and c2.size == self.cfg.size, f"{path}: kinds or input size differ"
            m2 = SpikePercept(c2).to(device).eval()
            m2.load_state_dict(s2["model"])
            self.models.append((m2, c2))
        self.device = device
        self.index = {k: i for i, k in enumerate(self.cfg.kinds)}
        self.last_cm = None
        self.oracle = {ROLES[r] for r in os.environ.get("PERCEPT_ORACLE", "").split(",") if r}      # diagnosis only: nodes of these roles keep their true position
        self.rigid = os.environ.get("PERCEPT_RIGID", "0") == "1"     # refit each rigid object's pose to its estimated nodes
        # diagnosis only: "k,b" moves every object node along the head camera's ray by k * table offset + b (cm) when the
        # table height is randomized (the network's height error follows -k * offset - b, scripts/rt_view_height_check.py)
        tz = os.environ.get("PERCEPT_TABLE_ORACLE", "")
        self.table_oracle = tuple(float(v) for v in tz.split(",")) if tz else None
        # height from the wrist cameras' side view (scripts/rt_wrist_height_check.py): over the first WRIST_PLANS plans of
        # an episode, every object node whose estimate falls inside a wrist image is triangulated from its head and wrist
        # pixels; the median of (triangulated - estimated height) then moves every object node along the head ray
        self.wrist_z = os.environ.get("PERCEPT_WRIST_Z", "0") == "1"
        self.wrist_plans = int(os.environ.get("PERCEPT_WRIST_PLANS", "3"))
        self._ep, self._dz, self._plans = None, [], 0
        self.enc = None
        if getattr(self.cfg, "text_tokens", False):              # the instruction's token states, encoded once per instruction
            from ..text import TextEncoder
            self.enc = TextEncoder(device, pool="tokens", encoder="spikeminilm16", max_tokens=64)      # as scripts/train_perception.TEXT_TOKENS

    @staticmethod
    def model_points(sk):
        """(n objects nodes, 4): each node's model-frame point relative to its object's box centre and a valid flag, for
        objects with box faces (the training side, scripts/train_perception.model_frame_points, uses the same rule)."""
        geo = np.zeros((len(sk.slot) - N_ROBOT, 4), np.float32)
        for s in np.unique(sk.slot[N_ROBOT:]):
            j = np.flatnonzero(sk.slot == s)
            if not (sk.role[j] == ROLES["face"]).any() or np.isnan(sk.local[j]).any():
                continue
            c = sk.local[j][sk.role[j] == ROLES["center"]][0]
            geo[j - N_ROBOT, :3] = sk.local[j] - c
            geo[j - N_ROBOT, 3] = 1.0
        return geo

    @staticmethod
    def box_size(sk, s):
        """Half-extents of slot s's box from its face nodes' model-frame points, or None."""
        j = np.flatnonzero(sk.slot == s)
        f = {int((sk.node_id[i] - N_ROBOT) % SLOT_SIZE): sk.local[i] for i in j}
        if not all(k in f for k in range(1, 7)) or any(np.isnan(f[k]).any() for k in range(1, 7)):
            return None
        return np.array([np.linalg.norm(f[2 * k + 1] - f[2 * k + 2]) / 2 for k in range(3)])

    @staticmethod
    def refit(est, sk, skip=()):
        """Robust rigid fit per object: the pose that maps the model-frame points (sk.local) onto the estimates
        (Kabsch, Huber reweighting at 2 cm), then every node of the object from that pose. Slots in skip keep their estimates."""
        out = est.copy()
        for s in np.unique(sk.slot[N_ROBOT:]):
            if s in skip:
                continue
            j = np.flatnonzero(sk.slot == s)
            P = sk.local[j]
            if np.isnan(P).any() or len(j) < 3:
                continue
            Q, w = est[j].astype(np.float64), np.ones(len(j))
            for _ in range(4):
                pc, qc = (w[:, None] * P).sum(0) / w.sum(), (w[:, None] * Q).sum(0) / w.sum()
                U, _, Vt = np.linalg.svd(((P - pc) * w[:, None]).T @ (Q - qc))
                D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
                R = Vt.T @ D @ U.T
                F = (P - pc) @ R.T + qc
                w = 1.0 / np.maximum(np.linalg.norm(F - Q, axis=1) / 0.02, 1.0)
            out[j] = F
        return out

    def wrist_height(self, sk, out, mats, obj, margin=16.0):
        """obj (n, 3) fused object nodes; returns them moved along the head ray by the episode's wrist-view height correction."""
        if self._ep is not sk:                                   # a new episode (one RTSkeleton per episode)
            self._ep, self._dz, self._plans = sk, [], 0
        M = np.asarray(mats, np.float64)
        cen = [-np.linalg.solve(m[:, :3], m[:, 3]) for m in M]
        if self._plans < self.wrist_plans:
            bins = (torch.arange(out["logit_u"][0].shape[-1], device=self.device).float() + 0.5) / 2
            uv = [torch.stack([(out["logit_u"][v][0].float().softmax(-1) * bins).sum(-1), (out["logit_v"][v][0].float().softmax(-1) * bins).sum(-1)], -1).cpu().numpy()
                  for v in range(len(M))]
            ray = lambda v, px: (lambda d: d / np.linalg.norm(d))(np.linalg.solve(M[v][:, :3], np.array([px[0], px[1], 1.0])))
            for j, p in enumerate(obj.astype(np.float64)):
                for v in range(1, len(M)):
                    h = M[v][:, :3] @ p + M[v][:, 3]
                    if h[2] <= 0:
                        continue
                    px = h[:2] / h[2]
                    if not (margin < px[0] < 256 - margin and margin < px[1] < 256 - margin):
                        continue
                    d1, d2, w = ray(0, uv[0][j]), ray(v, uv[v][j]), cen[0] - cen[v]
                    b, d, e = d1 @ d2, d1 @ w, d2 @ w
                    den = 1.0 - b * b
                    if den < 1e-9:
                        continue
                    s, t = (b * e - d) / den, (e - b * d) / den
                    if s <= 0 or t <= 0:
                        continue
                    q = (cen[0] + s * d1 + cen[v] + t * d2) / 2
                    if abs(q[2] - p[2]) < 0.06:
                        self._dz.append(q[2] - p[2])
            self._plans += 1
        if len(self._dz) < 3:
            return obj
        dz = float(np.median(self._dz))
        C, p = cen[0], obj.astype(np.float64)
        t = (p[:, 2] + dz - C[2]) / np.where(np.abs(p[:, 2] - C[2]) < 1e-6, 1e-6, p[:, 2] - C[2])
        return (C + t[:, None] * (p - C)).astype(obj.dtype)

    @torch.no_grad()
    def __call__(self, task, sk, skel, text):
        """skel (n, 3) world frame with exact arm nodes; text (1, 1, D); returns a copy with estimated object nodes."""
        images, mats = views(task)
        self.last_views = (images, mats)
        x = torch.as_tensor(images, device=self.device).float().permute(0, 3, 1, 2) / 255
        x = F.interpolate(x, size=(self.cfg.size, self.cfg.size), mode="bilinear", antialias=True)
        as_long = lambda v: torch.as_tensor(np.asarray(v[N_ROBOT:], np.int64), device=self.device)[None]
        g0, g1 = GRIP_NODES
        n_f = len(FINGERS)
        width = lambda g: np.linalg.norm(skel[g + 1] - skel[g + n_f])
        grip = torch.as_tensor(np.r_[skel[g0], width(g0), skel[g1], width(g1)], dtype=torch.float32, device=self.device)[None]
        kinds = [self.index.get(re.sub(r"_\d+$", "", sk.object_names[s - 1]), 0) for s in sk.slot[N_ROBOT:]]
        # slots of identical objects (two breads): numbered left to right, no shape query, no refit; with dup_shape only
        # same-kind slots of the same box size (as scripts/train_perception.identical_groups), else every same-kind slot
        slot_kind = {s: k for s, k in zip(sk.slot[N_ROBOT:], kinds)}
        shared = set()
        if getattr(self.cfg, "dup_order", False):
            for a, ka in slot_kind.items():
                for b, kb in slot_kind.items():
                    if a != b and ka == kb:
                        if getattr(self.cfg, "dup_shape", False):
                            sa, sb = self.box_size(sk, a), self.box_size(sk, b)
                            if sa is not None and sb is not None and np.abs(sa - sb).max() >= 0.003:
                                continue
                        shared.add(a)
        geo = None
        if getattr(self.cfg, "geo", False):
            g = self.model_points(sk)
            g[np.isin(sk.slot[N_ROBOT:], list(shared))] = 0.0
            geo = torch.as_tensor(g, device=self.device)[None]
        tok = tok_mask = None
        if self.enc is not None:
            tok, tok_mask = self.enc([str(task.get_instruction())])
        fused = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for m, c in self.models:
                o = m(((x - 0.5) / 0.25)[None], torch.as_tensor(mats, device=self.device)[None], text[:, 0].float(), as_long(sk.node_id), as_long(sk.role), grip,
                      torch.as_tensor(kinds, device=self.device)[None], geo if getattr(c, "geo", False) else None,
                      tok if getattr(c, "text_tokens", False) else None, tok_mask if getattr(c, "text_tokens", False) else None)
                fused.append(o["fused"][0].float())
                if len(fused) == 1:
                    out = o
        est = skel.copy()
        est[N_ROBOT:] = (fused[0] if len(fused) == 1 else torch.stack(fused).mean(0)).cpu().numpy()
        if self.table_oracle is not None and getattr(task, "random_table_height", 0) > 0:
            k, b = self.table_oracle
            off = float(task.table_z_bias)
            off -= -0.1 if off < -0.05 else 0.0                      # place_dual_shoes lowers its table by a fixed 10 cm
            dz = (k * off * 100 + b) / 100
            M = np.asarray(mats[0], np.float64)
            C = -np.linalg.solve(M[:, :3], M[:, 3])                  # head camera centre
            p = est[N_ROBOT:].astype(np.float64)
            t = (p[:, 2] + dz - C[2]) / np.where(np.abs(p[:, 2] - C[2]) < 1e-6, 1e-6, p[:, 2] - C[2])
            est[N_ROBOT:] = (C + t[:, None] * (p - C)).astype(est.dtype)
        if self.wrist_z:
            est[N_ROBOT:] = self.wrist_height(sk, out, mats, est[N_ROBOT:])
        if self.rigid:
            est = self.refit(est, sk, shared)
        if self.oracle:
            keep = np.isin(sk.role, list(self.oracle))
            est[keep] = skel[keep]
        self.last_cm = float(np.linalg.norm(est[N_ROBOT:] - skel[N_ROBOT:], axis=-1).mean() * 100)
        return est
