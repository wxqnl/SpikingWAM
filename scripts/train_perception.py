"""Train the spiking keypoint front end (spikewam/perception.py) on the frames of scripts/perception_cache.py.

The last 10% of the demonstrations of every task are held out. The score is the distance in cm between the estimated and
the true object nodes in the robot base frame.

  CUDA_VISIBLE_DEVICES=4 python scripts/train_perception.py --size 160 --tag s160
  robust front end: --root data/perception_rs,data/perception_dr --corrupt --paraphrase --camera --kinds --tag dr160
"""

import argparse
import glob
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from spikewam.perception import GRIP, IMAGE, N_OBJ, N_ROBOT, TIPS, PerceptConfig, SpikePercept, percept_loss, project, save_config
from spikewam.text import TextEncoder

TEXT_TOKENS = 64        # instruction tokens the front end's TextRead sees (the longest RoboTwin instruction has 52)

SWAPS = [("pick up", ["grasp", "lift", "take", "grab", "retrieve", "collect", "get"]),
         ("put", ["place", "set", "position", "move", "transfer", "deposit", "lay"]),
         ("place", ["put", "set", "position", "deposit", "lay"]),
         ("open", ["pull open", "slide open", "unlatch"]), ("close", ["shut", "push closed", "push shut"]),
         ("turn on", ["switch on", "activate", "start"]), ("turn off", ["switch off", "deactivate", "stop"]),
         ("push", ["slide", "shove", "move"]),
         ("black bowl", ["dark bowl", "dark round bowl", "dark container", "black dish", "dark vessel"]),
         ("bowl", ["dish", "round container", "basin"]), ("plate", ["dish", "flat dish", "dinner plate", "platter"]),
         ("ramekin", ["small baking dish", "small bowl", "little pot", "small round dish"]),
         ("stove", ["cooktop", "burner", "cooker", "hob", "range"]),
         ("wooden cabinet", ["wood cabinet", "wooden cupboard", "timber cabinet", "wooden chest"]),
         ("cabinet", ["cupboard", "storage unit", "chest"]), ("drawer", ["compartment", "sliding drawer", "pull-out drawer"]),
         ("mug", ["cup", "coffee cup"]), ("basket", ["bin", "hamper", "wicker container"]), ("tray", ["serving tray", "platter"]),
         ("microwave", ["microwave oven", "oven"]), ("wine bottle", ["bottle of wine", "dark bottle"]), ("wine rack", ["bottle rack", "wine holder"]),
         ("cookie box", ["box of cookies", "biscuit box", "cookie carton"]), ("book", ["hardcover", "volume"]), ("caddy", ["organizer", "desk organizer"]),
         ("moka pot", ["coffee maker", "stovetop coffee pot"]), ("frying pan", ["skillet", "pan"]),
         ("next to", ["beside", "adjacent to", "near", "by"]), ("on top of", ["atop", "above", "over"]),
         ("between", ["in between", "situated between", "located between", "in the space between"]),
         (" in the ", [" inside the ", " within the ", " into the "]), (" on the ", [" onto the ", " on top of the ", " upon the "]),
         ("and place it", ["then place it", "and then put it", "and set it", "and move it"]),
         ("table center", ["middle of the table", "center of the table", "table's middle"]),
         ("front", ["near side", "forward part"]), ("back", ["rear", "far side"]), ("top", ["upper", "uppermost"]), ("bottom", ["lower", "lowest"])]
FRAMES = ["{}", "{}", "please {}", "{} now", "your task is to {}", "the robot should {}", "can you {}", "carefully {}", "i need you to {}"]


def paraphrases(text, n, rng):
    """The instruction and n rewordings by word substitution. Our own vocabulary; the LIBERO-Plus test sentences are never read."""
    out = [text]
    for _ in range(n):
        s = text
        for a, bs in SWAPS:
            if a in s and rng.random() < 0.7:
                s = re.sub(re.escape(a), lambda _m: rng.choice(bs), s)
        out.append(rng.choice(FRAMES).format(s))
    return out


class Frames(torch.utils.data.Dataset):
    """Images are read from the memory-mapped files; everything else lives in tensors held by Data."""

    def __init__(self, rgb, task, local):
        self.rgb, self.task, self.local = rgb, task, local

    def __len__(self):
        return len(self.task)

    def __getitem__(self, i):
        return i, np.array(self.rgb[self.task[i]][self.local[i]])


class Data:
    def __init__(self, root, device, encoder, fold=0, folds=0, n_paraphrase=0, kinds=False):
        """root: comma list. folds > 0: demonstrations with index % folds == fold are held out, for out-of-fold estimates; else the last 10%."""
        roots = root.split(",")
        dirs = [d for r in roots for d in sorted(glob.glob(os.path.join(r, "*"))) if os.path.exists(os.path.join(d, "meta.npz"))]
        base = lambda d: re.sub(r"_s\d+$", "", os.path.basename(d))
        kind_of = {}
        for d in dirs:
            z = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
            if "kind" in z.files:
                kind_of[base(d)] = [str(k) for k in z["kind"]]
        if kinds:
            vocab = ["<unk>"] + sorted({k for v in kind_of.values() for k in v})
            self.kinds = tuple(vocab)
            index = {k: i for i, k in enumerate(vocab)}
        rgb, task, local, skel, mask, M, ids, roles, kind, grip, val, langs, src = [], [], [], [], [], [], [], [], [], [], [], [], []
        for k, d in enumerate(dirs):
            z = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
            rgb.append(np.load(os.path.join(d, "rgb.npy"), mmap_mode="r"))
            s, n = z["skel"], z["skel"].shape[1] - N_ROBOT
            assert n <= N_OBJ
            pad = np.zeros((len(s), N_OBJ, 3), np.float32)
            pad[:, :n] = s[:, N_ROBOT:]
            skel.append(pad)
            mask.append(np.tile(np.arange(N_OBJ) < n, (len(s), 1)))
            ids.append(np.tile(np.r_[z["node_id"][N_ROBOT:], np.zeros(N_OBJ - n, np.int64)], (len(s), 1)))
            roles.append(np.tile(np.r_[z["role"][N_ROBOT:], np.zeros(N_OBJ - n, np.int64)], (len(s), 1)))
            if kinds:
                kind.append(np.tile(np.r_[[index[x] for x in kind_of[base(d)]], np.zeros(N_OBJ - n, np.int64)], (len(s), 1)))
            M.append(z["M"])
            grip.append(np.c_[s[:, GRIP], np.linalg.norm(s[:, TIPS[0]] - s[:, TIPS[1]], axis=-1)])
            eps = z["ep"]
            uniq = np.unique(eps)
            cut = uniq[-max(1, round(len(uniq) * 0.1)):]         # shards hold every other demonstration
            val.append(eps % folds == fold if folds else np.isin(eps, cut))
            task.append(np.full(len(s), k))
            local.append(np.arange(len(s)))
            langs.append(str(z["language"]))
            src.append(np.full(len(s), next(i for i, r in enumerate(roots) if d.startswith(r.rstrip("/") + os.sep))))
        self.names = [os.path.basename(d) for d in dirs]
        self.rgb, self.task_np, self.local = rgb, np.concatenate(task), np.concatenate(local)
        cat = lambda x, t: torch.as_tensor(np.concatenate(x), dtype=t, device=device)
        self.task, self.skel, self.mask = cat(task, torch.long), cat(skel, torch.float32), cat(mask, torch.bool)
        self.M, self.ids, self.roles, self.grip = cat(M, torch.float32), cat(ids, torch.long), cat(roles, torch.long), cat(grip, torch.float32)
        self.kind = cat(kind, torch.long) if kinds else None
        is_val, source = np.concatenate(val), np.concatenate(src)
        self.train_idx, self.val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        self.val_by_root = [np.flatnonzero(is_val & (source == i)) for i in range(len(roots))]
        self.roots = [os.path.basename(r.rstrip("/")) for r in roots]
        enc = TextEncoder(device, pool="mean", encoder=encoder)
        rng = random.Random(0)
        bank = [paraphrases(x, n_paraphrase, rng) if n_paraphrase else [x] for x in langs]
        flat = sorted({s for b in bank for s in b})
        emb = enc(flat)[0][:, 0].float()
        where = {s: i for i, s in enumerate(flat)}
        self.text = torch.stack([emb[[where[s] for s in b]] for b in bank])       # (dirs, variants, D); variant 0 is the instruction
        print(f"{len(dirs)} task dirs, {len(self.train_idx)} train and {len(self.val_idx)} held-out frames; {self.text.shape[1]} instruction variants; {len(flat)} sentences", flush=True)

    def loader(self, idx, batch, workers, steps=None):
        ds = Frames(self.rgb, self.task_np[idx], self.local[idx])
        sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=batch * steps) if steps else None
        return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers, persistent_workers=workers > 0,
                                           prefetch_factor=4 if workers else None, drop_last=bool(steps))


SAME_SHAPE = 0.003       # same-kind objects whose box half-extents agree within this (m) are the same model (two identical breads)


def identical_groups(pos, ids, kind, same_shape):
    """RoboTwin, one frame's object nodes (positions, ids, kinds): groups of slots that are the same kind and, with
    same_shape, the same box size; only such slots cannot be told apart in an image (bottles of one kind but three models can)."""
    from spikewam.robotwin.layout import N_ROBOT as NR, SLOT_SIZE
    slot, within = (ids - NR) // SLOT_SIZE, (ids - NR) % SLOT_SIZE
    slots = list(np.unique(slot))
    kind_of = {sl: kind[np.flatnonzero(slot == sl)[0]] for sl in slots}

    def size(sl):
        f = {int(w): pos[j] for j, w in zip(np.flatnonzero(slot == sl), within[slot == sl])}
        return np.array([np.linalg.norm(f[2 * k + 1] - f[2 * k + 2]) / 2 for k in range(3)]) if all(k in f for k in range(1, 7)) else None

    groups = []
    for k in dict.fromkeys(kind_of.values()):
        rest = [sl for sl in slots if kind_of[sl] == k]
        while rest:
            a, rest = rest[0], rest[1:]
            g = [a]
            if same_shape:
                sa = size(a)
                match = [b for b in rest if sa is None or size(b) is None or np.abs(size(b) - sa).max() < SAME_SHAPE]
            else:
                match = list(rest)
            g += match
            rest = [b for b in rest if b not in match]
            if len(g) > 1:
                groups.append(g)
    return groups


def same_kind_order(z, same_shape=False):
    """RoboTwin: slots of identical objects (two breads; identical_groups) are numbered by the environment's creation order,
    which no image shows. Renumber them left to right by their x at the episode's first frame, kept for the whole episode.
    Returns the node order (frames, nodes), the node ids in that order, the number of episodes changed, and a (frames,
    nodes) mask of the nodes of such slots in the original order."""
    from spikewam.robotwin.layout import N_ROBOT as NR, SLOT_SIZE
    s, ids, kind, n_nodes, ep = z["skel"], z["node_id"], z["kind"], z["n_nodes"], z["ep"]
    order, new_ids, changed = np.tile(np.arange(s.shape[1]), (len(s), 1)), ids.copy(), 0
    dup = np.zeros(ids.shape, bool)
    for e in np.unique(ep):
        f = np.flatnonzero(ep == e)
        f0, n = f[0], int(n_nodes[f[0]])
        slot = (ids[f0, NR:n] - NR) // SLOT_SIZE
        slots = list(np.unique(slot))
        new_of = {sl: sl for sl in slots}
        for group in identical_groups(s[f0, NR:n], ids[f0, NR:n], kind[f0, NR:n], same_shape):
            by_x = sorted(group, key=lambda sl: s[f0, NR + np.flatnonzero(slot == sl), 0].mean())
            new_of.update({old: new for new, old in zip(sorted(group), by_x)})
            dup[np.ix_(f, NR + np.flatnonzero(np.isin(slot, group)))] = True
        if all(new_of[sl] == sl for sl in slots):
            continue
        changed += 1
        old_of = {v: k for k, v in new_of.items()}
        nodes = [NR + j for new in sorted(old_of) for j in np.flatnonzero(slot == old_of[new])]
        order[f, NR:n] = nodes
        within = (ids[f0, nodes] - NR) % SLOT_SIZE
        new_slot = np.concatenate([np.full(int((slot == old_of[new]).sum()), new) for new in sorted(old_of)])
        new_ids[np.ix_(f, np.arange(NR, n))] = NR + new_slot * SLOT_SIZE + within
    return order, new_ids, changed, dup


def model_frame_points(skel, ids, roles, mask):
    """RoboTwin, (N, n, 4): each object node's point in its object's model frame relative to the box centre, and a valid
    flag. The frame comes from the box-face nodes (+x, -x, +y, -y, +z, -z), i.e. the object model's geometry, the same
    numbers the closed loop reads from the model (RTSkeleton.local); objects without face nodes get zeros."""
    from spikewam.robotwin.layout import MAX_SLOTS, N_ROBOT as NR, SLOT_SIZE
    N, n, _ = skel.shape
    out = torch.zeros(N, n, 4, device=skel.device)
    slot, pos = ((ids - NR) // SLOT_SIZE).clamp(0, MAX_SLOTS - 1), (ids - NR) % SLOT_SIZE
    E = torch.eye(3, device=skel.device)
    for s in range(MAX_SLOTS):
        inslot = mask & (slot == s)
        pts, has = [], []
        for p in range(7):                                  # box centre, then the six faces
            sel = inslot & (pos == p)
            pts.append((skel * sel[..., None]).sum(1)); has.append(sel.any(1))
        ok = torch.stack(has, 1).all(1)
        if not ok.any():
            continue
        c = pts[0][ok]
        F = torch.stack(pts[1:], 1)[ok] - c[:, None]                                       # (k, 6, 3)
        h = torch.stack([(F[:, 2 * k] - F[:, 2 * k + 1]).norm(dim=-1) / 2 for k in range(3)], 1)
        P = torch.stack([sg * h[:, k:k + 1] * E[k] for k in range(3) for sg in (1, -1)], 1)  # the faces in the model frame
        U, _, Vt = torch.linalg.svd((P.transpose(1, 2) @ F).double())
        d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
        R = (Vt.transpose(1, 2) @ D @ U.transpose(1, 2)).float()                          # model frame -> world
        rel = (skel[ok] - c[:, None]) @ R                                                   # R^T (x - c)
        sel = inslot[ok]
        sub = out[ok]
        sub[..., :3] = torch.where(sel[..., None], rel, sub[..., :3])
        sub[..., 3] = torch.where(sel, torch.ones_like(sub[..., 3]), sub[..., 3])
        out[ok] = sub
    return out


class DataRT:
    """The frames of spikewam/robotwin/perception_cache.py; the same attributes as Data. Node layout and instruction change per episode; task holds the episode's text group."""

    def __init__(self, root, device, encoder, dup_order=False, dup_shape=False):
        from spikewam.robotwin.layout import FINGERS, GRIP_NODES, N_ROBOT as NR, SLOT_SIZE
        roots = root.split(",")
        dirs = [d for r in roots for d in sorted(glob.glob(os.path.join(r, "*"))) if os.path.exists(os.path.join(d, "meta.npz"))]
        metas = [np.load(os.path.join(d, "meta.npz"), allow_pickle=True) for d in dirs]
        self.kinds = tuple(["<unk>"] + sorted({str(k) for z in metas for k in z["vocab"]} - {"robot"}))
        index = {k: i for i, k in enumerate(self.kinds)}
        n_obj = max(z["skel"].shape[1] - NR for z in metas)
        rgb, task, local, skel, mask, M, ids, roles, kind, grip, val, groups, src = [], [], [], [], [], [], [], [], [], [], [], [], []
        sentences, base, reordered, dup = [], 0, 0, []
        for k, (d, z) in enumerate(zip(dirs, metas)):
            z = dict(z.items())
            order, new_ids, changed, same = same_kind_order(z, dup_shape)
            if dup_order and changed:
                for key in ("skel", "role", "kind"):
                    z[key] = np.take_along_axis(z[key], order[..., None] if z[key].ndim == 3 else order, 1)
                z["node_id"] = new_ids
                same = np.take_along_axis(same, order, 1)
                reordered += changed
            # nodes of identical objects (two breads), whose slot number no image shows
            dup.append(np.concatenate([same[:, NR:], np.zeros((len(same), n_obj - (same.shape[1] - NR)), bool)], 1))
            rgb.append(np.load(os.path.join(d, "rgb.npy"), mmap_mode="r"))
            s, n = z["skel"], z["skel"].shape[1] - NR
            pad = lambda a, dtype: np.concatenate([a, np.zeros((len(a), n_obj - n) + a.shape[2:], dtype)], 1)
            skel.append(pad(s[:, NR:], np.float32))
            mask.append(np.arange(n_obj)[None] < (z["n_nodes"][:, None] - NR))
            ids.append(pad(z["node_id"][:, NR:], np.int64))
            roles.append(pad(z["role"][:, NR:], np.int64))
            local_vocab = np.array([index.get(str(v), 0) for v in z["vocab"]])
            kind.append(pad(local_vocab[z["kind"][:, NR:]], np.int64) * mask[-1])
            M.append(z["M"])
            g0, g1 = GRIP_NODES
            n_f = len(FINGERS)
            width = lambda g: np.linalg.norm(s[:, g + 1] - s[:, g + n_f], axis=-1)[:, None]
            grip.append(np.c_[s[:, g0], width(g0), s[:, g1], width(g1)])
            eps = z["ep"]
            uniq = np.unique(eps)
            val.append(np.isin(eps, uniq[-max(1, round(len(uniq) * 0.1)):]))
            task.append(np.full(len(s), k))
            local.append(np.arange(len(s)))
            groups.append(base + eps)
            base += len(z["language"])
            sentences += [list(map(str, t)) for t in z["language"]]
            src.append(np.full(len(s), next(i for i, r in enumerate(roots) if d.startswith(r.rstrip("/") + os.sep))))
        self.names = [os.path.basename(d) for d in dirs]
        self.rgb, self.task_np, self.local = rgb, np.concatenate(task), np.concatenate(local)
        cat = lambda x, t: torch.as_tensor(np.concatenate(x), dtype=t, device=device)
        self.task, self.skel, self.mask = cat(groups, torch.long), cat(skel, torch.float32), cat(mask, torch.bool)
        self.M, self.ids, self.roles, self.grip, self.kind = cat(M, torch.float32), cat(ids, torch.long), cat(roles, torch.long), cat(grip, torch.float32), cat(kind, torch.long)
        is_val, source, cfgs = np.concatenate(val), np.concatenate(src), np.array([n.split("_")[1] for n in self.names])
        self.train_idx, self.val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        self.source = source
        closed = np.array(["rtcl" in os.path.basename(r.rstrip("/")) for r in roots])[source]       # frames of the policy's own closed loop
        self.roots = ["clean", "randomized"]
        self.val_by_root = [np.flatnonzero(is_val & ~closed & (cfgs[self.task_np] == c)) for c in self.roots]
        if closed.any():
            self.roots.append("closed_loop")
            self.val_by_root.append(np.flatnonzero(is_val & closed))
        enc = TextEncoder(device, pool="mean", encoder=encoder)
        flat = sorted({t for b in sentences for t in b})
        emb = torch.cat([enc(flat[i:i + 2048])[0][:, 0].float() for i in range(0, len(flat), 2048)])
        where = {t: i for i, t in enumerate(flat)}
        self.text = torch.stack([emb[[where[t] for t in b]] for b in sentences])
        self.text_id = torch.as_tensor([[where[t] for t in b] for b in sentences], device=device)
        self.flat, self.n_obj = flat, n_obj
        self.dup = torch.as_tensor(np.concatenate(dup), device=device)
        print(f"{len(dirs)} sets, {len(self.train_idx)} train and {len(self.val_idx)} held-out frames; {base} episodes, {len(flat)} sentences, {n_obj} object nodes at most, {len(self.kinds)} kinds"
              + (f"; same-kind slots renumbered left to right in {reordered} episodes" if dup_order else ""), flush=True)

    def load_tokens(self, device, encoder):
        """The token states of every instruction (for a model with text_tokens), bf16 on the device."""
        enc = TextEncoder(device, pool="tokens", encoder=encoder, max_tokens=TEXT_TOKENS)
        toks, masks = [], []
        for i in range(0, len(self.flat), 2048):
            t, m = enc(self.flat[i:i + 2048])
            toks.append(t.to(torch.bfloat16))
            masks.append(m)
            enc.cache.clear()
        self.tokens, self.tok_mask = torch.cat(toks), torch.cat(masks)
        print(f"token states {tuple(self.tokens.shape)}, {float(self.tok_mask.sum(1).float().mean()):.1f} tokens per instruction, "
              f"{float((self.tok_mask[:, -1]).float().mean()) * 100:.1f}% fill all {self.tok_mask.shape[1]}", flush=True)

    def tokens_of(self, group, variant):
        if getattr(self, "tokens", None) is None:
            return None, None
        j = self.text_id[group, variant]
        return self.tokens[j], self.tok_mask[j]

    loader = Data.loader


def _conv(x, kernels):
    """Per-image convolution: x (N, 3, H, W), kernels (N, K, K)."""
    n, k = x.shape[0], kernels.shape[-1]
    w = kernels[:, None].repeat_interleave(3, 0)
    y = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode="replicate")
    return F.conv2d(y.reshape(1, 3 * n, *y.shape[2:]), w, groups=3 * n).reshape(n, 3, *x.shape[2:])


def corrupt(x):
    """Common image corruptions (ImageNet-C families) with per-image strengths; x (N, 3, 256, 256) in [0, 1]."""
    n, dev = x.shape[0], x.device
    if n == 0:
        return x
    op = torch.randint(0, 6, (n,), device=dev)
    out = x.clone()
    r = lambda lo, hi, k: lo + (hi - lo) * torch.rand(k, device=dev)
    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, x.shape[3], device=dev), torch.linspace(-1, 1, x.shape[2], device=dev), indexing="xy"), -1)
    for o in range(6):
        i = torch.nonzero(op == o)[:, 0]
        k = len(i)
        if k == 0:
            continue
        a = x[i]
        if o == 0:                    # gaussian blur
            s = r(0.5, 10, k)
            t = torch.arange(-31, 32, device=dev).float()
            g = torch.exp(-0.5 * (t[None] / s[:, None]) ** 2)
            g = g / g.sum(1, keepdim=True)
            a = _conv(a, g[:, None] * g[:, :, None])
        elif o == 1:                  # motion blur
            length, ang = r(5, 41, k), r(0, math.pi, k)
            t = torch.arange(-24, 25, device=dev).float()
            yy, xx = torch.meshgrid(t, t, indexing="ij")
            along = xx[None] * ang.cos()[:, None, None] + yy[None] * ang.sin()[:, None, None]
            across = -xx[None] * ang.sin()[:, None, None] + yy[None] * ang.cos()[:, None, None]
            w = ((along.abs() <= length[:, None, None] / 2) & (across.abs() <= 0.6)).float()
            a = _conv(a, w / w.sum((1, 2), keepdim=True).clamp_min(1))
        elif o == 2:                  # zoom blur
            m = r(0.05, 0.6, k)
            acc = 0
            for z in torch.linspace(0, 1, 8, device=dev):
                acc = acc + F.grid_sample(a, grid[None] / (1 + z * m)[:, None, None, None], align_corners=True, padding_mode="border")
            a = acc / 8
        elif o == 3:                  # fog
            c = r(0.3, 5, k)[:, None, None, None]
            p = sum(F.interpolate(torch.rand(k, 1, s, s, device=dev), size=x.shape[2:], mode="bicubic", align_corners=False) * 0.5 ** e for e, s in enumerate((4, 8, 16, 32)))
            p = p / p.flatten(1).amax(1).clamp_min(1e-3)[:, None, None, None]
            top = a.flatten(1).amax(1)[:, None, None, None]
            a = (a + c * p) * top / (top + c)
        elif o == 4:                  # glass: local pixel displacement and a slight blur
            d = r(1, 5, k)[:, None, None, None] / (x.shape[2] / 2)
            jitter = (torch.rand(k, *x.shape[2:], 2, device=dev) * 2 - 1) * d
            a = F.grid_sample(a, grid[None] + jitter, align_corners=True, padding_mode="border")
            t = torch.arange(-3, 4, device=dev).float()
            g = torch.exp(-0.5 * (t[None] / r(0.4, 2.0, k)[:, None]) ** 2)
            g = g / g.sum(1, keepdim=True)
            a = _conv(a, g[:, None] * g[:, :, None])
        else:                         # sensor noise
            a = a + r(0.02, 0.2, k)[:, None, None, None] * torch.randn_like(a)
        out[i] = a.clamp(0, 1).to(out.dtype)
    return out


def prepare(images, size, device, augment, corruptions=False):
    x = images.to(device, non_blocking=True).float().permute(0, 1, 4, 2, 3) / 255          # (B, views, 3, H, W)
    b, views = x.shape[:2]
    if augment and corruptions:
        agent = x[:, 0]
        i = torch.nonzero(torch.rand(b, device=device) < 0.5)[:, 0]
        agent[i] = corrupt(agent[i])
    x = F.interpolate(x.flatten(0, 1), size=(size, size), mode="bilinear", antialias=True)
    if augment:
        n = x.shape[0]
        r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, 1, 1, 1, device=device)
        gray = x.mean(1, keepdim=True)
        x = gray + r(0.6, 1.4) * (x - gray)                                                 # saturation
        mean = x.mean((1, 2, 3), keepdim=True)
        x = mean + r(0.7, 1.3) * (x - mean)                                                 # contrast
        x = x * r(0.7, 1.3)                                                                 # brightness
        x = x + r(0.0, 0.02) * torch.randn_like(x)
        x = x.clamp(0, 1)
    return ((x - 0.5) / 0.25).unflatten(0, (b, views))


def geom_aug(x, M, strength):
    """Random zoom (1 +- strength) and shift (+- strength of the half width) of every view, with the same change of its camera matrix, so
    that a node's pixel follows the image. x (B, V, 3, S, S) normalized; M (B, V, 3, 4) on the IMAGE grid. A node pushed out of the frame counts as unseen."""
    b, v, s = x.shape[0], x.shape[1], x.shape[-1]
    n, dev = b * v, x.device
    a = 1 + strength * (2 * torch.rand(n, device=dev) - 1)
    t = strength * (2 * torch.rand(n, 2, device=dev) - 1)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1] = 1 / a, 1 / a
    theta[:, 0, 2], theta[:, 1, 2] = -t[:, 0] / a, -t[:, 1] / a
    grid = F.affine_grid(theta, (n, 3, s, s), align_corners=False)
    y = F.grid_sample(x.flatten(0, 1), grid, mode="bilinear", padding_mode="border", align_corners=False).unflatten(0, (b, v))
    c = IMAGE / 2
    a, t = a.unflatten(0, (b, v)), t.unflatten(0, (b, v))
    M2 = M.clone()
    M2[:, :, 0] = a[..., None] * M[:, :, 0] + ((1 - a) * c + t[..., 0] * c)[..., None] * M[:, :, 2]
    M2[:, :, 1] = a[..., None] * M[:, :, 1] + ((1 - a) * c + t[..., 1] * c)[..., None] * M[:, :, 2]
    return y, M2


@torch.no_grad()
def evaluate(model, data, idx, a, device, full=False):
    model.eval()
    if not full:
        idx = idx[np.linspace(0, len(idx) - 1, 6000).astype(int)]
    views = model.cfg.views
    names = ("agent", "wrist") if views == 2 else ("head", "left", "right")
    errs, by_view, vis_all = [], [[] for _ in range(views)], [[] for _ in range(views)]
    ld = data.loader(idx, 256, 4)
    for k, images in ld:
        i = torch.as_tensor(idx[k.numpy()], device=device)
        kind = data.kind[i] if data.kind is not None else None
        tok, tok_mask = data.tokens_of(data.task[i], 0) if hasattr(data, "tokens_of") else (None, None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(prepare(images, a.size, device, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], kind,
                        data.geo[i] if getattr(data, "geo", None) is not None else None, tok, tok_mask)
        skel, mask = data.skel[i], data.mask[i]
        errs.append((out["fused"] - skel).norm(dim=-1)[mask].cpu())
        for v in range(views):
            uv, d = project(data.M[i][:, v], skel)
            vis = mask & (d > 0) & (uv >= 0).all(-1) & (uv < IMAGE).all(-1)
            by_view[v].append((out["xyz"][v] - skel).norm(dim=-1)[vis].cpu())
            vis_all[v].append(vis[mask].float().cpu())
    model.train()
    q = lambda x: {"mean": float(x.mean() * 100), "p50": float(x.median() * 100), "p90": float(x.quantile(0.9) * 100), "within_1cm": float((x < 0.01).float().mean()), "within_2cm": float((x < 0.02).float().mean())}
    e = torch.cat(errs)
    res = {"fused": q(e), "n": len(e)}
    for v, name in enumerate(names):
        res[name], res["seen_" + name] = q(torch.cat(by_view[v])), float(torch.cat(vis_all[v]).mean())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rs"), help="comma list of frame caches")
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--widths", default="24,48,96,192")
    p.add_argument("--depths", default="1,2,3")
    p.add_argument("--attn", type=int, default=2)
    p.add_argument("--out_channels", type=int, default=128)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--text_encoder", default="spikeminilm16")
    p.add_argument("--init", default="", help="checkpoint of an unfinished run with the same settings; continues at its step")
    p.add_argument("--load", default="", help="start from these weights with a fresh schedule (fine-tuning)")
    p.add_argument("--workspace", type=float, default=0.0, help="RoboTwin: leave out of the loss object nodes farther than this (m, horizontal) from the table centre or below the floor (knocked away in a failed closed loop)")
    p.add_argument("--repeat", default="", help="comma list root_index:times, RoboTwin: draw the training frames of these roots this many times as often")
    p.add_argument("--kind_weight", default="", help="comma list kind:weight, loss weight of the nodes of these object kinds")
    p.add_argument("--tag", required=True)
    p.add_argument("--folds", type=int, default=0, help="hold out one of this many folds of the demonstrations")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--corrupt", action="store_true", help="common image corruptions on half of the third-person views")
    p.add_argument("--paraphrase", type=int, default=0, help="rewordings per instruction, sampled at training")
    p.add_argument("--camera", action="store_true", help="the camera matrix enters the query")
    p.add_argument("--kinds", action="store_true", help="the object kind of each slot enters the query")
    p.add_argument("--bench", default="libero", choices=("libero", "robotwin"))
    p.add_argument("--geom", type=float, default=0.0, help="random zoom and shift of the views with the matching camera matrices; the strength")
    p.add_argument("--geo", action="store_true", help="RoboTwin: each node's point in its object's model frame enters the query")
    p.add_argument("--hidden", type=int, default=128, help="neurons between the query map and the coordinate bins")
    p.add_argument("--text_tokens", action="store_true", help="the node query also reads the instruction's token states (TextRead)")
    p.add_argument("--dup_order", action="store_true", help="RoboTwin: slots of one kind numbered left to right at the episode start; their geometry query left out")
    p.add_argument("--dup_shape", action="store_true", help="with --dup_order: only same-kind slots of the same box size (identical objects); other same-kind objects keep their numbers and geometry")
    a = p.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    out = os.path.join(os.environ.get("SW_RUNS", "runs"), "perception", a.tag)
    os.makedirs(out, exist_ok=True)
    bench = {}
    if a.bench == "robotwin":
        from spikewam.robotwin.layout import N_NODE_IDS as RT_IDS, N_ROLES as RT_ROLES
        data = DataRT(a.root, device, a.text_encoder, a.dup_order, a.dup_shape)
        bench = dict(views=3, n_ids=RT_IDS, n_roles=RT_ROLES, grip_dim=8, z_lo=-0.05, z_hi=1.30, bins_z=270)        # the floor to above the table, 5 mm bins
        if a.geo:
            data.geo = model_frame_points(data.skel, data.ids, data.roles, data.mask)
            bench["geo"] = True
            if a.dup_order:                       # which of two breads a slot holds is a position, not a shape: no model points
                data.geo[data.dup] = 0.0
                print(f"geometry query left out for {int((data.dup & data.mask).sum())} nodes of same-kind slots", flush=True)
        if a.text_tokens:
            data.load_tokens(device, a.text_encoder)
            bench["text_tokens"] = True
        if a.dup_order:
            bench["dup_order"] = True
            bench["dup_shape"] = a.dup_shape
        if a.workspace:
            inside = (data.skel[..., :2].norm(dim=-1) < a.workspace) & (data.skel[..., 2] > -0.1)
            print(f"workspace {a.workspace} m: {int((data.mask & ~inside).sum())} of {int(data.mask.sum())} object nodes left out of the loss", flush=True)
            data.mask &= inside
    else:
        assert not a.geo, "--geo is RoboTwin only"
        data = Data(a.root, device, a.text_encoder, a.fold, a.folds, a.paraphrase, a.kinds)
    cfg = PerceptConfig(size=a.size, widths=tuple(int(x) for x in a.widths.split(",")), depths=tuple(int(x) for x in a.depths.split(",")), attn=a.attn, out=a.out_channels, hidden=a.hidden, camera=a.camera, kinds=data.kinds if a.kinds else (), **bench)
    model = SpikePercept(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_param / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))) * 0.98 + 0.02))
    start = 0
    if a.load:
        st = torch.load(a.load, map_location=device, weights_only=False)
        assert tuple(st["cfg"].get("kinds", ())) == tuple(cfg.kinds), "object kinds differ from the loaded model"
        missing, unexpected = model.load_state_dict(st["model"], strict=False)
        assert not unexpected and all(k.startswith("read.") for k in missing), (missing, unexpected)
        print(f"weights from {a.load} (step {st['step']})" + (f"; new: {len(missing)} tensors of the token reader (output at zero)" if missing else ""), flush=True)
    if a.init:                                # carry on a run from its checkpoint: its weights and schedule position, a fresh optimizer
        st = torch.load(a.init, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        start = st["step"]
        for _ in range(start):
            sched.step()
        print(f"continuing from step {start}, lr {sched.get_last_lr()[0]:.2e}", flush=True)
    kind_w = None
    if a.kind_weight:
        kind_w = torch.ones(len(data.kinds), device=device)
        for item in a.kind_weight.split(","):
            name, value = item.split(":")
            kind_w[data.kinds.index(name)] = float(value)
    train_idx = data.train_idx
    for item in filter(None, a.repeat.split(",")):
        r, times = (int(v) for v in item.split(":"))
        extra = train_idx[data.source[train_idx] == r]
        train_idx = np.concatenate([train_idx] + [extra] * (times - 1))
        print(f"root {r}: {len(extra)} training frames drawn {times} times as often", flush=True)
    log, t0, ld = [], time.time(), data.loader(train_idx, a.batch, a.workers, a.steps - start)
    for step, (k, images) in enumerate(ld, start + 1):
        i = torch.as_tensor(train_idx[k.numpy()], device=device)
        x = prepare(images, a.size, device, True, a.corrupt)
        M = data.M[i]
        if a.geom:
            x, M = geom_aug(x, M, a.geom)
        tid = data.task[i]
        variant = torch.randint(0, data.text.shape[1], tid.shape, device=device)
        text = data.text[tid, variant]
        tok, tok_mask = data.tokens_of(tid, variant) if hasattr(data, "tokens_of") else (None, None)
        kind = data.kind[i] if data.kind is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x, M, text, data.ids[i], data.roles[i], data.grip[i], kind, data.geo[i] if a.geo else None, tok, tok_mask)
        loss, parts = percept_loss(o, M, data.skel[i], data.mask[i], kind_w[kind] if kind_w is not None else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {float(loss):.3f} " + " ".join(f"{n} {v:.3f}" for n, v in parts.items()) + f" {(step - start) / (time.time() - t0):.1f} it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            res = {r: evaluate(model, data, v, a, device, full=step == a.steps) for r, v in zip(data.roots, data.val_by_root)}
            res["step"] = step
            log.append(res)
            print(json.dumps(res), flush=True)
            json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
            torch.save({"model": model.state_dict(), "cfg": save_config(cfg), "step": step, "n_param": n_param}, os.path.join(out, "ckpt.pt"))


if __name__ == "__main__":
    main()
