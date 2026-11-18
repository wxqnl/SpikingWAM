"""GPU-resident skeleton dataset: LIBERO (output of scripts/extract_skeletons.py) or RoboTwin 2.0 (output of
spikewam/robotwin/collect.py; every episode is its own entry, with its own nodes and instruction, because the
objects of a RoboTwin task change between episodes).

A training sample at step t of a demo holds the current skeleton S_t, the future skeletons S_{t+d_k} for
d_k = FUTURE_STRIDE * k (k = 1..H_s, clamped to the last step of the demo) and the action chunk a_t..a_{t+H_a-1}
(clamped the same way, i.e. the last action repeats). Coordinates are normalised per axis with training-set mean and
std over all nodes and steps (one shift and scale per axis keeps the geometry between nodes); the six motion
dimensions of the action are mapped from their 1st-99th percentile range to [-1, 1]; the gripper stays at -1 / +1.
RoboTwin actions (joint positions of both arms, grippers included, at the next frame) are all motion dimensions.

posture_rel (RoboTwin, implementation record 4.13): the arm joints of the chunk are given relative to the current joint
vector q_t (= the action of the previous frame; at an episode's first frame the first action stands in for it), divided
by the standard deviation of a_{t+k,j} - q_{t,j} over the training set (no clipping; a percentile range would be set
by the rare fast motions), per chunk step k and joint j (posture_scale "step") or per joint over the whole chunk
("joint": the loss then weighs the steps of a joint in the same physical units, as the absolute chunk does); the
grippers stay absolute. The network then never has to carry the absolute posture, which its integer spikes keep only
to about 2 degrees (scripts/rt_posture_toy.py); q_t is added back outside the network.

proprio (RoboTwin, record 4.13): the current skeleton gets one more channel, the measured joint angle of each arm link
node (link j <- joint j, grip site <- gripper; normalised as the absolute action but not clipped; 0 on every other
node), so that the network sees the posture its chunk is relative to (the wrist roll is barely visible in the node
positions). With posture_rel or proprio, training skips each episode's first frame (no measured q_0 in the data).

Corrective perturbation (RoboTwin, posture_rel, training only, set_perturb; record 4.13): a posture-relative chunk is
"q_t + learned change", and every training state lies on a demonstration, so the network never learns to pull a
displaced arm back (closed loop it carries any offset through the episode). Per sample and per arm whose gripper is open
and stays open over the next K steps (the return ends before a demonstrated grasp), in a fraction p of the samples, the
arm joints are displaced, q~ = q_t + eps, eps ~ N(0, sigma) per joint; the robot nodes of the current skeleton move by
FK(q~) - FK(q_t) (spikewam/robotwin/fk.py, fitted to the data), the proprio channel and the anchor read q~, the chunk
returns to the demonstration over K steps, a_{t+k} + max(0, 1 - (k+1)/K) eps, and the future skeleton at offset d keeps
the share max(0, 1 - d/K) of the displacement (FK at the demonstrated joints of that frame). Arms holding an object are
not displaced (its nodes would have to move with the hand).

CALVIN (bench "calvin", output of scripts/calvin_extract.py): one entry per language-annotated segment, with its own
instruction; an entry carries frames past the segment's end (the recording continues) that chunks and future skeletons
read but samples never start at (n_sample). Instructions repeat across segments: `languages` is the list of distinct
ones and get()'s "text" indexes it.

RoboTwin data may come from several collections (root = comma list of directories, e.g. demo_clean and
demo_randomized; the FK parameters are read from the first). instr="seen": every sample reads one of its episode's
seen instructions (RoboTwin draws 100 per episode), drawn anew each time, instead of always the first; `languages` is
then the list of distinct instructions and get()'s "text" indexes it (otherwise "text" = "task", one per episode).
"""

from __future__ import annotations

import glob
import json
import math
import os
import re

import numpy as np
import torch

from .libero_tasks import SUITES

N_MAX = 40          # node slots per skeleton (the largest LIBERO task has 35 nodes)
ACTION_DIM = 7
MOTION_DIMS = 6
ROBOTWIN = {"action_dim": 14, "motion_dims": 14}   # node slots: the largest scene, rounded up to 8
RT_ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]    # arm joints of the RoboTwin action (6 and 13: grippers)
RT_PROPRIO_NODES = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17]   # node of each action dimension (layout.py)


class Normalizer:
    def __init__(self, stats):
        self.stats = stats
        self.pos_mean = np.asarray(stats["pos_mean"], np.float32)
        self.pos_std = np.asarray(stats["pos_std"], np.float32)
        self.act_lo = np.asarray(stats["act_q01"], np.float32)
        self.act_hi = np.asarray(stats["act_q99"], np.float32)
        self.md = len(self.act_lo)
        self.rel_scale = np.asarray(stats["rel_scale"], np.float32) if "rel_scale" in stats else None   # (H, D)

    @classmethod
    def fit(cls, tasks, motion_dims=MOTION_DIMS):
        pos = np.concatenate([t["skel"].reshape(-1, 3) for t in tasks])
        act = np.concatenate([t["actions"] for t in tasks])
        return cls({"pos_mean": pos.mean(0).tolist(), "pos_std": pos.std(0).tolist(),
                    "act_q01": np.quantile(act[:, :motion_dims], 0.01, axis=0).tolist(),
                    "act_q99": np.quantile(act[:, :motion_dims], 0.99, axis=0).tolist()})

    def pos(self, x):
        return (x - self.pos_mean) / self.pos_std

    def pos_inv(self, x):
        return x * self.pos_std + self.pos_mean

    def act(self, a):
        a = a.copy()
        m = a[..., :self.md]
        a[..., :self.md] = np.clip(2 * (m - self.act_lo) / (self.act_hi - self.act_lo) - 1, -1, 1)
        return a

    def act_inv(self, a, q=None):
        """Normalised chunk (H, D) back to actions; posture-relative arm joints need the current joint vector q."""
        out = a.copy()
        out[..., :self.md] = (a[..., :self.md] + 1) / 2 * (self.act_hi - self.act_lo) + self.act_lo
        out[..., self.md:] = np.where(a[..., self.md:] > 0, 1.0, -1.0)
        if self.rel_scale is not None:
            out[..., RT_ARM] = a[..., RT_ARM] * self.rel_scale[:a.shape[-2], RT_ARM] + q[RT_ARM]
        return out

    def joints(self, q):
        """Joint vector (torch or numpy, (..., D)) normalised as the action, without clipping (proprio input)."""
        lo, hi = self.act_lo, self.act_hi
        if torch.is_tensor(q):
            lo, hi = torch.as_tensor(lo, device=q.device), torch.as_tensor(hi, device=q.device)
        return 2 * (q - lo) / (hi - lo) - 1

    def units(self, horizon):
        """(H, D) action units per normalised unit (radians for RoboTwin joints)."""
        u = np.tile((self.act_hi - self.act_lo) / 2, (horizon, 1))
        if self.rel_scale is not None:
            u[:, RT_ARM] = self.rel_scale[:horizon, RT_ARM]
        return u


def load_tasks(root, suites=SUITES):
    tasks = []
    for s in suites:
        for path in sorted(glob.glob(os.path.join(root, s, "*.npz"))):
            z = np.load(path)
            tasks.append({k: z[k] for k in z.files} | {"suite": s, "task_id": int(os.path.basename(path)[:2])})
            tasks[-1]["language"] = str(tasks[-1]["language"])
    return tasks


def load_robotwin(root, task_names=None, seen=False):
    """One entry per episode of every <root>/<task>.npz (spikewam/robotwin/collect.py); root may be a comma list of
    directories. seen: also keep each episode's seen instructions."""
    entries = []
    for r in root.split(","):
        for path in sorted(glob.glob(os.path.join(r, "*.npz"))):
            name = os.path.basename(path)[:-4]
            if task_names and name not in task_names:
                continue
            # every z[key] of an NpzFile decompresses the whole array again, and an episode's slice would keep that
            # copy alive: read each array once (500-episode tasks otherwise take hours and hundreds of GB per rank)
            with np.load(path, allow_pickle=True) as f:
                z = {k: f[k] for k in f.files}
            for e, (st, ln, n) in enumerate(zip(z["ep_start"], z["ep_len"], z["n_nodes"])):
                entries.append({"skel": z["skel"][st:st + ln, :n], "actions": z["actions"][st:st + ln],
                                "node_id": z["node_id"][e, :n], "role": z["role"][e, :n], "slot": z["slot"][e, :n],
                                "ep_start": np.array([0]), "ep_len": np.array([ln]), "language": str(z["language"][e]),
                                "suite": name, "task_id": e})
                if seen:
                    entries[-1]["seen"] = json.loads(z["meta"][e])["seen"]
    return entries


def load_calvin(root):
    """One entry per language-annotated segment of every <root>/<scene>.npz (scripts/calvin_extract.py); root may be a
    comma list of directories."""
    entries = []
    for r in root.split(","):
        for path in sorted(glob.glob(os.path.join(r, "*.npz"))):
            with np.load(path) as f:
                z = {k: f[k] for k in ("skel", "actions", "ep_start", "ep_len", "n_sample", "language", "node_id",
                                       "role", "slot")}
            for e, (st, ln, ns) in enumerate(zip(z["ep_start"], z["ep_len"], z["n_sample"])):
                entries.append({"skel": z["skel"][st:st + ln], "actions": z["actions"][st:st + ln],
                                "node_id": z["node_id"], "role": z["role"], "slot": z["slot"],
                                "ep_start": np.array([0]), "ep_len": np.array([ln]), "n_sample": int(ns),
                                "language": str(z["language"][e]), "suite": os.path.basename(path)[:-4],
                                "task_id": e})
    return entries


def proprio_channel(q, n_max):
    """(B, n_max, 1): the normalised joint vector q (B, 14) on the arm link and grip-site nodes, 0 elsewhere."""
    ch = torch.zeros(q.shape[0], n_max, 1, dtype=q.dtype, device=q.device)
    ch[:, RT_PROPRIO_NODES, 0] = q
    return ch


def pad_nodes(n, arrays_and_fill, n_max=N_MAX):
    """Pad per-node descriptor arrays of length n to n_max."""
    return [np.concatenate([a, np.full(n_max - n, fill, a.dtype)]) for a, fill in arrays_and_fill]


class SkeletonDataset:
    def __init__(self, root, device, horizon=32, n_future=8, future_stride=4, suites=SUITES, normalizer=None,
                 bench="libero", posture_rel=False, posture_scale="step", proprio=False, instr="first"):
        assert not (posture_rel or proprio) or bench == "robotwin", "posture_rel, proprio: RoboTwin joint actions only"
        assert posture_scale in ("step", "joint") and instr in ("first", "seen")
        assert instr == "first" or bench == "robotwin", "instr=seen: RoboTwin episodes carry their seen instructions"
        assert bench in ("libero", "robotwin", "calvin")
        if bench == "robotwin":
            tasks = load_robotwin(root, seen=instr == "seen")
            n_max, motion_dims = -(-max(t["skel"].shape[1] for t in tasks) // 8) * 8, ROBOTWIN["motion_dims"]
        elif bench == "calvin":
            tasks = load_calvin(root)
            n_max, motion_dims = -(-max(t["skel"].shape[1] for t in tasks) // 8) * 8, MOTION_DIMS
        else:
            tasks = load_tasks(root, suites)
            n_max, motion_dims = N_MAX, MOTION_DIMS
        self.n_max = n_max
        self.norm = normalizer or Normalizer.fit(tasks, motion_dims)
        self.horizon, self.n_future, self.future_stride = horizon, n_future, future_stride
        self.languages = [t["language"] for t in tasks]
        self.instr_idx = None
        if instr == "seen":         # (episodes, n_seen) indices into the distinct instructions
            index = {}
            n_seen = max(len(t["seen"]) for t in tasks)
            rows = [[index.setdefault(t["seen"][k % len(t["seen"])], len(index)) for k in range(n_seen)]
                    for t in tasks]
            self.languages = list(index)
            self.instr_idx = torch.as_tensor(rows, dtype=torch.long, device=device)
        elif bench == "calvin":     # distinct instructions; each entry reads its own
            index = {}
            rows = [[index.setdefault(t["language"], len(index))] for t in tasks]
            self.languages = list(index)
            self.instr_idx = torch.as_tensor(rows, dtype=torch.long, device=device)
        self.task_names = [f"{t['suite']}/{t['task_id']}" for t in tasks]
        skel, act, act_raw, task_of, ep_start, ep_end, desc, start_ok = [], [], [], [], [], [], [], []
        skel_in = []                 # optional current-skeleton input (front-end estimates); targets stay true
        offset = 0
        for i, t in enumerate(tasks):
            n = t["skel"].shape[1]
            s = np.zeros((len(t["skel"]), n_max, 3), np.float32)
            s[:, :n] = self.norm.pos(t["skel"])
            skel.append(s)
            if "skel_in" in t:
                s_in = np.zeros_like(s)
                s_in[:, :n] = self.norm.pos(t["skel_in"])
                skel_in.append(s_in)
            act.append(self.norm.act(t["actions"]))
            act_raw.append(t["actions"].astype(np.float32))
            task_of.append(np.full(len(s), i))
            for st, ln in zip(t["ep_start"], t["ep_len"]):
                ep_start.append(np.full(ln, offset + st))
                ep_end.append(np.full(ln, offset + st + ln - 1))
                start_ok.append(np.arange(ln) < t.get("n_sample", ln))
            offset += len(s)
            node_id, role, slot, mask = pad_nodes(n, [(t["node_id"], 0), (t["role"], 0), (t["slot"], 0),
                                                      (np.ones(n, bool), False)], n_max)
            desc.append(np.stack([node_id, role, slot, mask]))
        to = lambda x, dt: torch.as_tensor(np.concatenate(x), dtype=dt, device=device)
        self.skel, self.act = to(skel, torch.float32), to(act, torch.float32)
        self.skel_in = to(skel_in, torch.float32) if skel_in else self.skel
        self.task_of = to(task_of, torch.long)
        self.ep_start, self.ep_end = to(ep_start, torch.long), to(ep_end, torch.long)
        self.motion_lag = 0          # set by the trainer (WAMConfig.motion_lag): append node displacements to "cur"
        desc = torch.as_tensor(np.stack(desc), device=device)            # (tasks, 4, N_MAX)
        self.node_id, self.role, self.slot = desc[:, 0].long(), desc[:, 1].long(), desc[:, 2].long()
        self.node_mask = desc[:, 3].bool()
        self.device = device
        self.n_steps = len(self.skel)
        # frames a sample may start at (CALVIN: inside the annotated segment); None: every frame
        self.starts = None if bench != "calvin" else torch.nonzero(to(start_ok, torch.bool))[:, 0]
        self.posture_rel, self.posture_scale, self.proprio = posture_rel, posture_scale, proprio
        if posture_rel or proprio:
            self.act_raw = to(act_raw, torch.float32)
            i = torch.arange(self.n_steps, device=device)
            self.anchor = torch.where(i > self.ep_start, i - 1, i)
            self.with_prev = torch.nonzero(i > self.ep_start)[:, 0]     # frames with a measured previous joint vector
        if posture_rel and self.norm.rel_scale is None:
            self.fit_rel_scale()
        self.root, self.perturb_sigma = root.split(",")[0], 0.0
        self.res = None

    def load_fk(self):
        """Forward kinematics of the robot nodes (spikewam/robotwin/fk.py, parameters fitted to this data)."""
        from .robotwin.fk import ArmFK
        with open(os.path.join(self.root, "fk_params.json")) as f:
            self.fk = ArmFK(json.load(f))
        self.pos_std_t = torch.as_tensor(self.norm.pos_std, device=self.device)

    def set_input_noise(self, path):
        """Front-end errors on the current skeleton: each sample's object nodes get the residuals of one random held-out frame of
        its task (scripts/rt_residuals.py). Targets stay true."""
        z = np.load(path, allow_pickle=True)
        index = {str(n): i for i, n in enumerate(z["tasks"])}
        self.res = torch.as_tensor(z["res"], dtype=torch.float32, device=self.device)
        self.res_count = torch.as_tensor(z["count"], device=self.device)
        self.res_task = torch.as_tensor([index[re.sub(r"_part.*$", "", n.split("/")[0])] for n in self.task_names], device=self.device)
        self.res_std = torch.as_tensor(self.norm.pos_std, dtype=torch.float32, device=self.device)

    def set_perturb(self, sigma_deg, p=0.5, k_return=8):
        """Corrective perturbation of sample()'s batches (see the module docstring)."""
        assert self.posture_rel, "corrective perturbation: posture-relative chunks only"
        self.load_fk()
        self.perturb_sigma, self.perturb_p = math.radians(sigma_deg), p
        k = torch.arange(self.horizon, device=self.device, dtype=torch.float32)
        self.perturb_lam = (1 - (k + 1) / k_return).clamp_min(0)

    def perturb(self, q, ka):
        """eps (B, 14) on the picked arms (the gripper open at t and over the return) of a fraction p of the samples."""
        eps = torch.zeros_like(q)
        k_return = int((self.perturb_lam > 0).sum()) + 1
        sample = torch.rand(len(q), device=q.device) < self.perturb_p
        for sl, g in ((slice(0, 6), 6), (slice(7, 13), 13)):
            pick = sample & (q[:, g] > 0.5) & (self.act_raw[ka[:, :k_return], g].min(1).values > 0.5)
            eps[:, sl] = torch.randn(len(q), 6, device=q.device) * self.perturb_sigma * pick[:, None]
        return eps

    def node_shift(self, q, eps):
        """Normalised displacement (..., 20, 3) of the robot nodes when the joints q (..., 14) move by eps."""
        qd, ed = q.reshape(-1, q.shape[-1]).double(), eps.reshape(-1, q.shape[-1]).double()
        d = (self.fk(qd + ed) - self.fk(qd)) / self.pos_std_t
        return d.float().reshape(*q.shape[:-1], *d.shape[1:])

    def fit_rel_scale(self, n=100000):
        """The standard deviation of a_{t+k} - q_t over n random training frames, per chunk step and joint (or per
        joint, broadcast over the steps)."""
        g = torch.Generator(device=self.device).manual_seed(0)
        i = torch.randint(0, self.n_steps, (n,), device=self.device, generator=g)
        ka = torch.minimum(i[:, None] + torch.arange(self.horizon, device=self.device), self.ep_end[i][:, None])
        d = self.act_raw[ka] - self.act_raw[self.anchor[i]][:, None]
        if self.posture_scale == "step":
            scale = d.std(0)
        else:
            scale = d.flatten(0, 1).std(0).expand(self.horizon, -1).contiguous()
        scale = scale.clamp_min(1e-3)
        self.norm.stats["rel_scale"] = scale.cpu().tolist()
        self.norm.rel_scale = scale.cpu().numpy()

    def sample(self, batch_size, generator=None):
        if self.posture_rel or self.proprio:
            pick = torch.randint(0, len(self.with_prev), (batch_size,), device=self.device, generator=generator)
            return self.get(self.with_prev[pick], perturb=bool(self.perturb_sigma))
        if self.starts is not None:
            pick = torch.randint(0, len(self.starts), (batch_size,), device=self.device, generator=generator)
            return self.get(self.starts[pick])
        i = torch.randint(0, self.n_steps, (batch_size,), device=self.device, generator=generator)
        return self.get(i)

    def get(self, i, perturb=False):
        end = self.ep_end[i][:, None]
        ka = torch.minimum(i[:, None] + torch.arange(self.horizon, device=self.device), end)
        ks = torch.minimum(i[:, None] + self.future_stride * torch.arange(1, self.n_future + 1, device=self.device),
                           end)
        task = self.task_of[i]
        cur = self.skel_in[i]
        if self.res is not None:
            rt = self.res_task[task]
            frame = (torch.rand(len(i), device=self.device) * self.res_count[rt]).long()
            r = self.res[rt, frame].gather(1, self.node_id[task][..., None].expand(-1, -1, 3))
            cur = cur + (r / self.res_std) * (self.node_mask[task] & (self.slot[task] > 0))[..., None]
        if self.motion_lag:          # displacement since motion_lag steps before (from the episode's first frame)
            cur = torch.cat([cur, cur - self.skel[torch.maximum(i - self.motion_lag, self.ep_start[i])]], -1)
        q = self.act_raw[self.anchor[i]] if (self.posture_rel or self.proprio) else None
        future = self.skel[ks]
        if perturb:
            eps = self.perturb(q, ka)
            d = self.node_shift(q, eps)
            cur[:, :d.shape[1], :3] += d
            share = (1 - self.future_stride * torch.arange(1, self.n_future + 1, device=self.device)
                     / (self.perturb_lam > 0).sum().add(1)).clamp_min(0)
            if share.any():          # future frame at offset d: the demonstrated joints there are those of frame d-1
                f = share > 0
                qf = self.act_raw[(ks[:, f] - 1).clamp_min(self.ep_start[i][:, None])]
                future[:, f, :d.shape[1], :3] += self.node_shift(qf, share[f][None, :, None] * eps[:, None])
            q = q + eps
        if self.proprio:
            cur = torch.cat([cur, proprio_channel(self.norm.joints(q), self.n_max)], -1)
        text = task
        if self.instr_idx is not None:
            pick = torch.randint(0, self.instr_idx.shape[1], task.shape, device=self.device)
            text = self.instr_idx[task, pick]
        out = {"cur": cur, "future": future, "actions": self.act[ka], "task": task, "text": text,
               "node_id": self.node_id[task], "role": self.role[task], "slot": self.slot[task],
               "node_mask": self.node_mask[task]}
        if self.posture_rel:
            target = self.act_raw[ka]
            if perturb:              # back to the demonstration over k_return steps
                target = target + self.perturb_lam[None, :, None] * eps[:, None]
            scale = torch.as_tensor(self.norm.rel_scale, device=self.device)
            out["actions"][..., RT_ARM] = ((target - q[:, None]) / scale)[..., RT_ARM]
            out["anchor"] = q
        return out

    def save_stats(self, path):
        with open(path, "w") as f:
            json.dump(self.norm.stats, f, indent=1)
