"""Coincidence attention (research plan section 4.4): a relative-geometry term for attention over skeleton nodes.

For head h and query anchor a_i (a node's own position, or the grip site for future-frame and action tokens),
    G^h_ij = sum_r g^l_{h,r} * < ring_r(a_i + Delta_h), ring_r(p_j) >,
where ring_r is one phase ring of the grid spike code and <.,.> counts coincident spikes (AND, then count). The
count on a ring depends only on the phase of u_d . (p_j - a_i - Delta_h), so G is translation-invariant by
construction. Delta_h (one 3D offset per head) says where the head looks relative to its anchor; g (one weight per
layer, head and ring) is learned and starts at zero, so a model starts without the geometry term.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .spiking import GridSpikeCode

GRIP_NODE = 8  # robot node order: links 1-7, hand, grip site, finger tips (spikewam.skeleton)


class CoincidenceGeometry(nn.Module):
    def __init__(self, heads, depth, shift=True, shift_init=0.03, **code_kwargs):
        super().__init__()
        self.code = GridSpikeCode(**code_kwargs)
        n_rings = self.code.n_dirs * self.code.n_scales
        self.heads = heads
        self.delta = nn.Parameter(torch.randn(heads, 3) * shift_init) if shift else None
        self.gain = nn.Parameter(torch.zeros(depth, heads, n_rings))

    def overlaps(self, anchors, keys):
        """Coincidence counts (B, H, Lq, Nk, rings) between query anchors (B, Lq, 3) and key positions (B, Nk, 3)."""
        q = anchors[:, None]
        if self.delta is not None:
            q = q + self.delta[None, :, None]
        else:
            q = q.expand(-1, self.heads, -1, -1)
        rq = self.code.rings(q)                 # (B, H, Lq, R, n)
        rk = self.code.rings(keys)              # (B, Nk, R, n)
        return torch.einsum("bhqrn,bkrn->bhqkr", rq, rk)

    def bias(self, ov, layer):
        """(B, H, Lq, Nk) attention bias of one layer."""
        return torch.einsum("bhqkr,hr->bhqk", ov, self.gain[layer].to(ov.dtype))
