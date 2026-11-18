"""Spiking building blocks for SpikeWAM.

Neuron: integer LIF in the I-LIF style of Spike-driven Transformer V3 (and of Spike-VLA's IntegerPLIF): one training
step emits s = clamp(round(u / theta), -D, D), which at inference is |s| unit spikes of one sign over D timesteps, so
every layer that reads s only accumulates its weight columns. Gradient: straight through inside [-D, D]. D plays the
role of the timestep count T.

Coordinate codes (research plan E1): every code maps metric positions (..., 3) in the robot frame to binary spikes.
  GridSpikeCode   ours: D_dir directions x M scales x N phase neurons; neuron n of ring (d, m) fires when the phase
                  (u_d . p / lambda_m) mod 1 lies within w/2N of n/N (circular distance).
  PlaceCode       single-scale population code per axis (PopSAN style): Gaussian tuning curves, thresholded.
  VoxelCode       E-3DSNN-style voxel code, factorised per axis (a joint 3D one-hot needs (1/v)^3 neurons).
  GrayCode        Gray code of the per-axis quantised coordinate (Gray-PE extended to metric coordinates).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# LIBERO workspace in the robot base frame (metres), padded; it bounds the single-scale codes.
WORKSPACE_LO = (-0.30, -0.60, -0.20)
WORKSPACE_HI = (1.10, 0.60, 0.80)


class IntSpike(nn.Module):
    """Integer (or bipolar integer) spikes with a learned threshold; counts spikes for the energy account.

    stochastic: in training, round stochastically and without bias (floor(x + U(0, 1)), so E[s] = x): the expected
    spike count then follows the current below one level, and the network can learn sub-level precision (with plain
    rounding the output is piecewise constant in the input and training settles on a coarser function).
    carry: at inference, None (plain rounding) or the remainder of the previous call (set to 0 to start): the neuron
    is not reset between calls but keeps what it did not fire (reset by subtraction), so over the flow steps of a
    plan its spikes add up to the summed current up to one level (sigma-delta)."""

    def __init__(self, levels=4, bipolar=True, threshold=0.5):
        super().__init__()
        self.levels, self.bipolar = int(levels), bipolar
        self.log_threshold = nn.Parameter(torch.tensor(math.log(threshold)))
        self.count = None  # set to a list to accumulate (spikes, neurons) per call
        self.stochastic = False
        self.carry = None

    def forward(self, current):
        x = current / self.log_threshold.exp()
        lo = -self.levels if self.bipolar else 0
        x = x.clamp(lo, self.levels)
        if self.stochastic and self.training:
            q = torch.floor(x.float() + torch.rand(x.shape, device=x.device)).to(x.dtype)
        elif self.carry is not None:
            v = x.float() + (self.carry if torch.is_tensor(self.carry) and self.carry.shape == x.shape else 0)
            q = torch.round(v).clamp(lo, self.levels)
            self.carry = (v - q).detach()
            q = q.to(x.dtype)
        else:
            q = torch.round(x)
        s = x + (q - x).detach()
        if self.count is not None:
            self.count.append((s.detach().abs().sum().item(), s.numel()))
        return s


def icosahedron_axes():
    """The 6 axes through opposite vertices of a regular icosahedron (unit vectors)."""
    g = (1 + 5 ** 0.5) / 2
    v = torch.tensor([[0, 1, g], [0, -1, g], [1, g, 0], [-1, g, 0], [g, 0, 1], [g, 0, -1]], dtype=torch.float32)
    return v / v.norm(dim=1, keepdim=True)


def fibonacci_axes(n):
    """n roughly uniform directions on the upper hemisphere (axes, sign-free)."""
    i = torch.arange(n, dtype=torch.float32) + 0.5
    z = i / n
    phi = math.pi * (1 + 5 ** 0.5) * i
    r = (1 - z ** 2).sqrt()
    return torch.stack([r * phi.cos(), r * phi.sin(), z], 1)


class _PhaseSpike(torch.autograd.Function):
    """Heaviside of (cos(angle) - cos(half window)); arctangent surrogate in the angle (for learned offsets)."""

    @staticmethod
    def forward(ctx, angle, half):
        ctx.save_for_backward(angle)
        ctx.half = half
        return (torch.cos(angle) >= math.cos(half)).to(angle.dtype)

    @staticmethod
    def backward(ctx, g):
        (angle,) = ctx.saved_tensors
        v = (torch.cos(angle) - math.cos(ctx.half)) / (1 - math.cos(ctx.half))
        return -g * torch.sin(angle) / (1 + (math.pi * v) ** 2) / (1 - math.cos(ctx.half)), None


class GridSpikeCode(nn.Module):
    def __init__(self, n_dirs=6, n_scales=12, n_phase=8, window=2, lambda_min=0.01, ratio=1.5):
        super().__init__()
        axes = icosahedron_axes() if n_dirs == 6 else fibonacci_axes(n_dirs)
        self.register_buffer("axes", axes)
        self.register_buffer("scales", lambda_min * ratio ** torch.arange(n_scales, dtype=torch.float32))
        self.register_buffer("phases", torch.arange(n_phase, dtype=torch.float32) / n_phase)
        self.n_dirs, self.n_scales, self.n_phase = n_dirs, n_scales, n_phase
        self.half = math.pi * window / n_phase
        self.dim = n_dirs * n_scales * n_phase

    def rings(self, p):
        """(..., n_dirs * n_scales, n_phase) binary code of positions p (..., 3)."""
        proj = p @ self.axes.T                                       # (..., D)
        phase = proj[..., :, None] / self.scales                     # (..., D, M)
        angle = 2 * math.pi * (phase[..., None] - self.phases)       # (..., D, M, N)
        return _PhaseSpike.apply(angle, self.half).flatten(-3, -2)

    def forward(self, p):
        return self.rings(p).flatten(-2)


class PlaceCode(nn.Module):
    def __init__(self, n_per_axis=192, width=1.5):
        super().__init__()
        lo, hi = torch.tensor(WORKSPACE_LO), torch.tensor(WORKSPACE_HI)
        self.register_buffer("centers", lo[:, None] + (hi - lo)[:, None] * (torch.arange(n_per_axis) + 0.5)
                             / n_per_axis)                           # (3, n)
        self.sigma = width * float((hi - lo).max()) / n_per_axis
        self.dim = 3 * n_per_axis

    def forward(self, p):
        d = (p[..., :, None] - self.centers) / self.sigma
        return (d.abs() <= 1.0).float().flatten(-2)


class VoxelCode(nn.Module):
    def __init__(self, voxel=0.01):
        super().__init__()
        self.register_buffer("lo", torch.tensor(WORKSPACE_LO))
        hi = torch.tensor(WORKSPACE_HI)
        self.voxel = voxel
        self.bins = [int(math.ceil(float(h - l) / voxel)) for l, h in zip(WORKSPACE_LO, WORKSPACE_HI)]
        self.dim = sum(self.bins)

    def forward(self, p):
        idx = ((p - self.lo) / self.voxel).floor().long()
        out = [torch.nn.functional.one_hot(idx[..., i].clamp(0, b - 1), b) for i, b in enumerate(self.bins)]
        return torch.cat(out, -1).float()


class GrayCode(nn.Module):
    def __init__(self, bits=12):
        super().__init__()
        self.register_buffer("lo", torch.tensor(WORKSPACE_LO))
        self.register_buffer("span", torch.tensor(WORKSPACE_HI) - torch.tensor(WORKSPACE_LO))
        self.bits = bits
        self.dim = 3 * bits

    def forward(self, p):
        q = ((p - self.lo) / self.span * (2 ** self.bits - 1)).round().long().clamp(0, 2 ** self.bits - 1)
        g = q ^ (q >> 1)
        shifts = torch.arange(self.bits, device=p.device)
        return ((g[..., None] >> shifts) & 1).flatten(-2).float()


CODES = {"grid": GridSpikeCode, "place": PlaceCode, "voxel": VoxelCode, "gray": GrayCode}
