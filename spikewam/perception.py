"""Spiking keypoint front end: two RGB views and the instruction to the object nodes of the skeleton.

The arm nodes come from the joint encoders and forward kinematics, so this network estimates the object nodes only. One
small spiking backbone reads both cameras. Every node asks a query (instruction, node id, role, gripper state) of the
feature map of each view; the query map goes through two spiking layers to three one-dimensional coordinate
distributions (pixel u, pixel v at half-pixel bins, height z at 5 mm bins), the coordinate classification of SimCC and
RTMPose, which needs no upsampling. The pixel is cast through the known camera onto the plane of height z, which gives
one 3D estimate per view, and a learned confidence per view fuses them.

All coordinates are in the robot base frame. M (B, 2, 3, 4) maps a base-frame point to pixels at 256 x 256 upright.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import N_NODE_IDS, N_ROBOT, N_ROLES, ROLES
from .spiking import IntSpike

N_OBJ = 24                                # the largest LIBERO task has 24 object nodes
IMAGE = 256                               # the pixel coordinates of M
UV_PER_PX = 2
BINS_UV = IMAGE * UV_PER_PX
Z_LO, Z_HI, BINS_Z = -0.15, 0.55, 140
GRIP, TIPS = 8, (9, 10)                   # skeleton indices of the grip site and the finger tips


@dataclass
class PerceptConfig:
    size: int = 160                       # network input, square
    widths: tuple = (24, 48, 96, 192)     # stem, stride 4, stride 8, stride 16
    depths: tuple = (1, 2, 3)             # blocks at stride 4, 8 and spike-driven attention keeps stride 16
    attn: int = 2                         # attention blocks at stride 16
    out: int = 128                        # feature channels at stride 8
    hidden: int = 128                     # neurons between the query map and the coordinate bins
    text_dim: int = 384
    camera: bool = False                  # the view's known camera matrix enters its query
    kinds: tuple = ()                     # object kinds of the skeleton slots, index 0 unknown; empty: no kind query
    geo: bool = False                     # the node's point in its object's model frame (from the object model) enters the query
    text_tokens: bool = False             # the node query also reads the instruction's token states (TextRead)
    dup_order: bool = False               # RoboTwin: slots of one kind are numbered left to right, without the geometry query
    dup_shape: bool = False               # with dup_order: only same-kind slots of the same box size (identical objects)
    views: int = 2                       # cameras read per frame; the rest of this block describes the benchmark's skeleton
    n_ids: int = N_NODE_IDS
    n_roles: int = N_ROLES
    grip_dim: int = 4                     # gripper state of the query: position and opening, per arm
    z_lo: float = Z_LO                    # range of the height bins, metres
    z_hi: float = Z_HI
    bins_z: int = BINS_Z


def spike():
    return IntSpike(levels=4, bipolar=False, threshold=1.0)


class ConvS(nn.Module):
    def __init__(self, a, b, k=3, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(a, b, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(b)
        self.spike = spike()

    def forward(self, x):
        return self.spike(self.bn(self.conv(x)))


class SepRes(nn.Module):
    """Depthwise and point convolutions with a spike shortcut."""

    def __init__(self, c, expand=2):
        super().__init__()
        self.depth = ConvS(c, c * expand, 3, g=c)
        self.point = ConvS(c * expand, c, 1)

    def forward(self, x):
        return x + self.point(self.depth(x))


class SpikeAttn(nn.Module):
    """Spike-driven self-attention on the stride-16 map: the global context that tells similar objects apart."""

    def __init__(self, c, heads=8):
        super().__init__()
        self.q, self.k, self.v = ConvS(c, c, 1), ConvS(c, c, 1), ConvS(c, c, 1)
        self.out = ConvS(c, c, 1)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        split = lambda t: t.flatten(2).transpose(1, 2).reshape(b, h * w, self.heads, c // self.heads).permute(0, 2, 1, 3)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        y = (q @ k.transpose(-1, -2) / c) @ v
        y = y.permute(0, 2, 1, 3).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        return x + self.out(y)


class TextRead(nn.Module):
    """Spike-driven cross-attention of the node queries on the instruction's token states, as SpikeAttn (spike Q, K, V,
    no softmax): which named object a slot is. The sentence mean barely tells "A left of B" from "B left of A" (cosine
    0.99 against 0.72 for another object). Queries and instruction are fixed within an episode, so this runs once per
    episode. The output layer starts at zero, so a loaded model starts unchanged."""

    def __init__(self, c, text_dim, heads=8):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(c, c), nn.Linear(text_dim, c), nn.Linear(text_dim, c)
        self.nq, self.nk, self.nv = nn.LayerNorm(c), nn.LayerNorm(c), nn.LayerNorm(c)
        self.sq, self.sk, self.sv = spike(), spike(), spike()
        self.out = nn.Linear(c, c)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.heads = heads

    def forward(self, q, tokens, mask):
        b, n, c = q.shape
        h, length = self.heads, tokens.shape[1]
        split = lambda t, m: t.reshape(b, m, h, c // h).transpose(1, 2)
        keep = mask[..., None].to(tokens.dtype)
        qs = split(self.sq(self.nq(self.q(q))), n)
        ks = split(self.sk(self.nk(self.k(tokens))) * keep, length)
        vs = split(self.sv(self.nv(self.v(tokens))) * keep, length)
        y = (qs @ ks.transpose(-1, -2) / c) @ vs
        return self.out(y.transpose(1, 2).reshape(b, n, c))


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c0, c1, c2, c3 = cfg.widths
        d4, d8, _ = cfg.depths
        self.stem = nn.Sequential(ConvS(3, c0, 3, 2), ConvS(c0, c1, 3, 2), *[SepRes(c1) for _ in range(d4)])
        self.s8 = nn.Sequential(ConvS(c1, c2, 3, 2), *[SepRes(c2) for _ in range(d8)])
        self.s16 = nn.Sequential(ConvS(c2, c3, 3, 2), *[SpikeAttn(c3) for _ in range(cfg.attn)])
        self.fuse = ConvS(c3 + c2, cfg.out, 1)
        self.refine = SepRes(cfg.out)

    def forward(self, x):
        f8 = self.s8(self.stem(x))
        f16 = F.interpolate(self.s16(f8), scale_factor=2, mode="nearest")
        return self.refine(self.fuse(torch.cat([f8, f16], 1)))


class ViewHead(nn.Module):
    def __init__(self, cells, hidden, bins_z=BINS_Z):
        super().__init__()
        self.norm = nn.LayerNorm(cells)
        self.map_spike = spike()
        self.mid = nn.Linear(cells, hidden)
        self.mid_norm = nn.LayerNorm(hidden)
        self.mid_spike = spike()
        self.out = nn.Linear(hidden, 2 * BINS_UV + bins_z + 1)

    def forward(self, feat, q):
        h = feat.flatten(2).transpose(1, 2) @ q.transpose(1, 2)            # (B, cells, N)
        s = self.map_spike(self.norm(h.transpose(1, 2)))
        return self.out(self.mid_spike(self.mid_norm(self.mid(s))))


def cast(M, u, v, z):
    """The point at height z on the ray of pixel (u, v); M (..., 3, 4), u, v, z (...), returns (..., 3)."""
    a11, a12 = M[..., 0, 0] - u * M[..., 2, 0], M[..., 0, 1] - u * M[..., 2, 1]
    a21, a22 = M[..., 1, 0] - v * M[..., 2, 0], M[..., 1, 1] - v * M[..., 2, 1]
    b1 = -((M[..., 0, 2] - u * M[..., 2, 2]) * z + M[..., 0, 3] - u * M[..., 2, 3])
    b2 = -((M[..., 1, 2] - v * M[..., 2, 2]) * z + M[..., 1, 3] - v * M[..., 2, 3])
    det = a11 * a22 - a12 * a21
    det = torch.where(det.abs() < 1e-3, torch.full_like(det, 1e-3), det)
    x, y = (b1 * a22 - b2 * a12) / det, (a11 * b2 - a21 * b1) / det
    return torch.stack([x.clamp(-1, 1), y.clamp(-1, 1), z], -1)


def project(M, p):
    """Pixels (..., 2) and depth (...) of base-frame points p (N, 3) with M (B, 3, 4)."""
    h = (M[:, None, :, :3] @ p[..., None])[..., 0] + M[:, None, :, 3]
    return h[..., :2] / h[..., 2:].clamp_min(1e-6), h[..., 2]


def soft_peak(logit, bins, radius=8):
    """Expected coordinate in a window around the highest bin: one candidate where the plain expectation falls between two."""
    p = logit.softmax(-1)
    i = torch.arange(logit.shape[-1], device=logit.device)
    q = p * ((i - p.argmax(-1, keepdim=True)).abs() <= radius)
    return (q * bins).sum(-1) / q.sum(-1).clamp_min(1e-9)


def cam_code(M):
    """The camera matrix (B, 3, 4) as 12 numbers of order one."""
    m = M.clone()
    m[:, :2] = m[:, :2] / IMAGE
    return m.flatten(1)


class SpikePercept(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.peak, self.wta = False, False        # inference only: decode at the peak; when the views disagree take the more confident one
        self.backbone = Backbone(cfg)
        self.node, self.role = nn.Embedding(cfg.n_ids, cfg.out), nn.Embedding(cfg.n_roles, cfg.out)
        self.text, self.grip = nn.Linear(cfg.text_dim, cfg.out), nn.Linear(cfg.grip_dim, cfg.out)
        self.kind = nn.Embedding(len(cfg.kinds), cfg.out) if cfg.kinds else None
        self.cam = nn.Linear(12, cfg.out) if cfg.camera else None
        self.geo = nn.Linear(4, cfg.out) if getattr(cfg, "geo", False) else None
        self.read = TextRead(cfg.out, cfg.text_dim) if getattr(cfg, "text_tokens", False) else None
        cells = (cfg.size // 8) ** 2
        self.heads = nn.ModuleList([ViewHead(cells, cfg.hidden, cfg.bins_z) for _ in range(cfg.views)])
        self.register_buffer("uv_bins", (torch.arange(BINS_UV).float() + 0.5) / UV_PER_PX, persistent=False)
        self.register_buffer("z_bins", cfg.z_lo + (torch.arange(cfg.bins_z).float() + 0.5) * (cfg.z_hi - cfg.z_lo) / cfg.bins_z, persistent=False)

    def forward(self, images, M, text, node_id, role, grip, kind=None, geo=None, tokens=None, tok_mask=None):
        """images (B, 2, 3, S, S) normalized; M (B, 2, 3, 4); text (B, D); node_id, role (B, N); grip (B, 4); kind (B, N);
        geo (B, N, 4): the node's model-frame point relative to its object's box centre (metres) and a valid flag;
        tokens (B, L, D), tok_mask (B, L): the instruction's token states, for a model with text_tokens."""
        b = images.shape[0]
        feat = self.backbone(images.flatten(0, 1))
        q = self.node(node_id) + self.role(role) + self.text(text)[:, None]
        if self.kind is not None:
            q = q + self.kind(kind)
        if self.geo is not None:
            q = q + self.geo(torch.cat([geo[..., :3] * 10, geo[..., 3:]], -1).to(q.dtype))
        if self.read is not None:
            q = q + self.read(q, tokens, tok_mask)
        q = q + self.grip(grip)[:, None]
        res = {"logit_u": [], "logit_v": [], "logit_z": [], "conf": [], "xyz": []}
        for v in range(self.cfg.views):
            qv = q if self.cam is None else q + self.cam(cam_code(M[:, v].float()))[:, None]
            o = self.heads[v](feat[v::self.cfg.views], qv).float()
            lu, lv, lz, conf = o[..., :BINS_UV], o[..., BINS_UV:2 * BINS_UV], o[..., 2 * BINS_UV:2 * BINS_UV + self.cfg.bins_z], o[..., -1]
            if self.peak:
                u, w, z = soft_peak(lu, self.uv_bins), soft_peak(lv, self.uv_bins), soft_peak(lz, self.z_bins)
            else:
                u, w = (lu.softmax(-1) * self.uv_bins).sum(-1), (lv.softmax(-1) * self.uv_bins).sum(-1)
                z = (lz.softmax(-1) * self.z_bins).sum(-1)
            res["xyz"].append(cast(M[:, v, None].float().expand(-1, q.shape[1], -1, -1), u, w, z))
            for k, x in zip(("logit_u", "logit_v", "logit_z", "conf"), (lu, lv, lz, conf)):
                res[k].append(x)
        conf = torch.stack(res["conf"], 0)
        xyz = torch.stack(res["xyz"], 0)
        res["fused"] = (xyz * conf.softmax(0)[..., None]).sum(0)
        if self.wta and self.cfg.views == 2:
            differ = (xyz[0] - xyz[1]).norm(dim=-1, keepdim=True) > 0.03
            res["fused"] = torch.where(differ, torch.where((conf[0] >= conf[1])[..., None], xyz[0], xyz[1]), res["fused"])
        res["b"], res["z_range"] = b, (self.cfg.z_lo, self.cfg.z_hi, self.cfg.bins_z)
        return res


def soft_ce(logit, target, bins, sigma):
    i = torch.arange(bins, device=logit.device, dtype=torch.float32)
    t = torch.exp(-0.5 * ((i - target[..., None]) / sigma) ** 2)
    t = t / t.sum(-1, keepdim=True).clamp_min(1e-9)
    return -(t * logit.log_softmax(-1)).sum(-1)


def percept_loss(out, M, skel, mask, node_w=None):
    """skel (B, N, 3) in metres; mask (B, N) bool; node_w (B, N) optional weight of each node. Returns the loss and a dict of its parts."""
    w = mask.float() if node_w is None else mask.float() * node_w
    z_lo, z_hi, bins_z = out["z_range"]
    z_idx = ((skel[..., 2] - z_lo) / (z_hi - z_lo) * bins_z - 0.5).clamp(0, bins_z - 1)
    loss_bins, loss_view, loss_conf = 0.0, 0.0, 0.0
    for v in range(len(out["xyz"])):
        uv, d = project(M[:, v].float(), skel)
        vis = mask & (d > 0) & (uv >= 0).all(-1) & (uv < IMAGE).all(-1)
        weight = vis.float() if node_w is None else vis.float() * node_w
        n = weight.sum().clamp_min(1)
        ce = soft_ce(out["logit_u"][v], uv[..., 0].clamp(0, IMAGE - 1) * UV_PER_PX - 0.5, BINS_UV, 2.0) \
            + soft_ce(out["logit_v"][v], uv[..., 1].clamp(0, IMAGE - 1) * UV_PER_PX - 0.5, BINS_UV, 2.0) \
            + soft_ce(out["logit_z"][v], z_idx, bins_z, 2.0)
        loss_bins = loss_bins + (ce * weight).sum() / n
        loss_view = loss_view + ((out["xyz"][v] - skel).abs().sum(-1) * 100 * weight).sum() / n
        reliable = vis & ((out["xyz"][v].detach() - skel).norm(dim=-1) < 0.03)      # the view can see the node and its estimate is right
        loss_conf = loss_conf + (F.binary_cross_entropy_with_logits(out["conf"][v], reliable.float(), reduction="none") * w).sum() / w.sum().clamp_min(1)
    fused = ((out["fused"] - skel).abs().sum(-1) * 100 * w).sum() / w.sum().clamp_min(1)
    total = loss_bins + 0.1 * loss_view + 0.3 * fused + loss_conf
    return total, {"bins": float(loss_bins), "view_cm": float(loss_view), "fused_cm": float(fused), "conf": float(loss_conf)}


def save_config(cfg):
    return asdict(cfg)


class FrontEnd:
    """The trained network in closed loop: renders both cameras and replaces the object nodes of a skeleton."""

    CAMS = ("agentview", "robot0_eye_in_hand")

    def __init__(self, ckpt, device):
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
        self.model = SpikePercept(self.cfg).to(device).eval()
        self.model.load_state_dict(st["model"])
        self.device = device
        self.last_cm = None
        self.corrupt = None                   # optional function of the raw third-person image (LIBERO-Plus sensor noise)
        self.smooth = 0                       # > 0: median of this many plans for objects away from the hand
        self.trace = False                    # keep per-node errors of each view and the view weights of the last call
        self.last_detail = None
        self.hist = {}
        self.hold = False                     # an object in the closed gripper is integrated in the hand frame
        self.held, self.rel = None, []
        self.text_enc = None

    def text_of(self, policy, instruction):
        """The instruction as the front end was trained to read it (scripts/train_perception: sentence mean of the spiking
        MiniLM). The final policy encodes it the same way, so its encoding is shared; a policy with another text encoder
        (the older models of the ablations) gets a separate encoding of the same sentence for the front end."""
        cfg = policy.model.cfg
        if getattr(cfg, "text_pool", "tokens") == "mean" and getattr(cfg, "text_encoder", None) == "spikeminilm16":
            return policy.text_tokens
        if self.text_enc is None:
            from .text import TextEncoder
            self.text_enc = TextEncoder(self.device, pool="mean", encoder="spikeminilm16")
        return self.text_enc([instruction])[0]

    def reset(self):
        """A new episode."""
        self.hist = {}
        self.held, self.rel = None, []

    def _hold(self, ext, est, skel):
        """An object between the closed fingers moves rigidly with the hand, whose pose the joint encoders give, so its
        nodes are the median of their estimates in the hand frame since the grasp. Uses only arm nodes and estimates."""
        tips = skel[TIPS[0]] - skel[TIPS[1]]
        if not 0.024 < np.linalg.norm(tips) < 0.088:          # open, or closed on nothing
            self.held, self.rel = None, []
            return
        p, z = skel[GRIP], skel[GRIP] - skel[GRIP - 1]
        z = z / np.linalg.norm(z)
        y = tips - tips.dot(z) * z
        y = y / np.linalg.norm(y)
        R = np.stack([np.cross(y, z), y, z], 1)
        obj, slots, center = est[N_ROBOT:], np.asarray(ext.slot[N_ROBOT:]), np.asarray(ext.role[N_ROBOT:]) == ROLES["center"]
        if self.held is None:
            reach = {s: np.linalg.norm(obj[(slots == s) & center][0] - p) for s in np.unique(slots[center])}
            s = min(reach, key=reach.get) if reach else None
            if s is None or reach[s] > 0.08:                  # a drawer handle is held far from its cabinet's center
                return
            self.held, self.rel = s, []
        nodes = slots == self.held
        rel = (obj[nodes] - p) @ R
        c = center[nodes]
        if self.rel and np.linalg.norm(np.median(np.stack(self.rel), 0)[c] - rel[c]) > 0.06:     # it left the hand
            self.held, self.rel = None, []
            return
        self.rel.append(rel)
        obj[nodes] = p + np.median(np.stack(self.rel), 0) @ R.T

    def _smooth(self, ext, est, grip):
        """An object away from the hand does not move, so its nodes are the median of its recent estimates; near the hand the history restarts."""
        obj, slots = est[N_ROBOT:], np.asarray(ext.slot[N_ROBOT:])
        for s in np.unique(slots):
            nodes = slots == s
            if np.linalg.norm(obj[nodes] - grip, axis=-1).min() < 0.12:
                self.hist[s] = []
                continue
            h = self.hist.setdefault(s, [])
            h.append(obj[nodes].copy())
            del h[:-self.smooth]
            obj[nodes] = np.median(np.stack(h), 0)

    def views(self, sim, ext):
        """Upright images (2, H, W, 3) uint8 and base-to-pixel matrices (2, 3, 4) of the simulator's current state."""
        from robosuite.utils.camera_utils import get_camera_transform_matrix
        data = sim.data._data
        R, p = data.xmat[ext.base].reshape(3, 3), data.xpos[ext.base]
        world_from_base = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
        images, mats = [], []
        for cam in self.CAMS:
            raw = sim.render(camera_name=cam, width=IMAGE, height=IMAGE, depth=False)
            if cam == "agentview" and self.corrupt is not None:
                raw = self.corrupt(raw)
            images.append(raw[::-1].copy())
            mats.append((get_camera_transform_matrix(sim, cam, IMAGE, IMAGE) @ world_from_base)[:3])
        return np.stack(images), np.array(mats, np.float32)

    @torch.no_grad()
    def __call__(self, sim, ext, skel, text):
        """skel (n, 3) in the base frame with exact arm nodes; text (1, 1, D); returns a copy with estimated object nodes."""
        images, mats = self.views(sim, ext)
        x = torch.as_tensor(images, device=self.device).float().permute(0, 3, 1, 2) / 255
        x = F.interpolate(x, size=(self.cfg.size, self.cfg.size), mode="bilinear", antialias=True)
        as_long = lambda v: torch.as_tensor(np.asarray(v[N_ROBOT:], np.int64), device=self.device)[None]
        grip = torch.as_tensor(np.r_[skel[GRIP], np.linalg.norm(skel[TIPS[0]] - skel[TIPS[1]])], dtype=torch.float32, device=self.device)[None]
        kind = None
        if self.cfg.kinds:
            index = {k: i for i, k in enumerate(self.cfg.kinds)}
            kind = torch.as_tensor([index.get(k, 0) for k in ext.kinds], device=self.device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(((x - 0.5) / 0.25)[None], torch.as_tensor(mats, device=self.device)[None], text[:, 0].float(), as_long(ext.node_id), as_long(ext.role), grip, kind)
        est = skel.copy()
        est[N_ROBOT:] = out["fused"][0].float().cpu().numpy()
        if self.smooth:
            self._smooth(ext, est, skel[GRIP])
        if self.hold:
            self._hold(ext, est, skel)
        self.last_cm =float(np.linalg.norm(est[N_ROBOT:] - skel[N_ROBOT:], axis=-1).mean() * 100)
        if self.trace:
            cm = lambda a: np.round(np.linalg.norm(a - skel[N_ROBOT:], axis=-1).astype(np.float64) * 100, 1).tolist()
            self.last_detail = {"fused": cm(est[N_ROBOT:]), "agent": cm(out["xyz"][0][0].float().cpu().numpy()), "wrist": cm(out["xyz"][1][0].float().cpu().numpy()),
                                "w": np.round(torch.stack([c.float() for c in out["conf"]], 0).softmax(0)[0, 0].cpu().numpy().astype(np.float64), 2).tolist(),
                                "hand": cm(skel[GRIP]), "twin": self._twins(sim, ext, est, skel)}
        return est

    @staticmethod
    def _twins(sim, ext, est, skel):
        """For each slot with another body of the same kind in the scene: cm from the estimated center to the true center and to the
        nearest such body, and from the true center to that body (diagnosis of confusing identical objects)."""
        import re
        data, model = sim.data._data, ext.model
        R, p = data.xmat[ext.base].reshape(3, 3), data.xpos[ext.base]
        names = [model.body(i).name for i in range(model.nbody)]
        slots, center, kinds, out = np.asarray(ext.slot[N_ROBOT:]), np.asarray(ext.role[N_ROBOT:]) == ROLES["center"], np.asarray(ext.kinds), {}
        for s in np.unique(slots[center]):
            i = np.flatnonzero((slots == s) & center)[0]
            kin = [b for b, n in enumerate(names) if re.fullmatch(re.escape(str(kinds[i])) + r"_\d+_main", n) and b != ext.groups[s - 1].root]
            if not kin:
                continue
            twins = (data.xpos[kin] - p) @ R
            e, t = est[N_ROBOT:][i], skel[N_ROBOT:][i]
            out[int(s)] = [round(float(x) * 100, 1) for x in (np.linalg.norm(e - t), np.linalg.norm(twins - e, axis=-1).min(), np.linalg.norm(twins - t, axis=-1).min())]
        return out


__all__ = ["SpikePercept", "PerceptConfig", "FrontEnd", "percept_loss", "project", "cast", "N_OBJ", "N_ROBOT", "GRIP", "TIPS", "IMAGE", "save_config"]
