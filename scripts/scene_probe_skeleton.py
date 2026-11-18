"""Skeleton-vs-camera probes, skeleton side: the final SpikeWAM policy on the scenes of scene_frames.py.

Inference as deployed on LIBERO (1 flow step, k = 1, one starting noise per task reused at every frame). Per frame:
  actions    env action chunks of the frame (twice in one batch, the second copy is the null intervention) and of
             every intervention of scene_frames.py; the skeleton holds only the robot and the task objects, so the
             light, tint and other-object interventions leave its input unchanged;
  attention  the action tokens' self-attention weight (Shiftmax with the coincidence bias, per head) on every
             skeleton node, averaged over layers, heads and action queries, summed per node group;
  events     with every frame encoded alone, in demo order (change-driven firing carries over between frames): every
             nn.Linear's input events per token, spike inputs as unit spikes (dense) or unit changes (delta), real
             inputs as MACs (all, or those of changed inputs), split into robot nodes, task-object nodes and action
             tokens (attention products and norms are not split).
World-model contact test (future branch, 10 Euler steps from the same noise; futures at +4 ... +32 steps), at the
frames where the demo closes the gripper on a task object (its centre node within 6 cm of the grip site):
  held       the scene as it is;
  open       the two finger-tip nodes moved 2 cm apart each, along the line between them (gripper opened);
  away       the held object's nodes moved 10 cm horizontally, away from the grip site.
Follow ratio of the held object at each future offset: (predicted object xy displacement . predicted grip-site xy
displacement) / |grip-site xy displacement|^2, over offsets where the grip site moves >= 1 cm in xy (1: the object
moves with the hand, 0: it stays). Also at frames where every task object is >= 15 cm from the grip site ("free").

  python scripts/scene_probe_skeleton.py --ckpt runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from spikewam.events import is_spike, stream_of
from spikewam.policy import SkeletonPolicy
from spikewam.skeleton import N_ROBOT, ROLES
from spikewam.spike_wam import shiftmax

GRIP, TIPS = 8, (9, 10)
E_SOP, E_MAC = 0.9e-12, 4.6e-12
GROUPS = ("robot_nodes", "task_object_nodes", "action_tokens")


class TokenEvents:
    def __init__(self, model):
        self.on, self.n, self.prev = False, 0, {}
        self.tot = {m: np.zeros(len(GROUPS)) for m in ("dense", "delta")}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and not name.split(".")[0] in ("t_emb", "t_mod", "final_mod") \
                    and not name.endswith("ckv"):
                mod.register_forward_pre_hook(lambda m, inp, name=name: self.linear(name, m, inp[0]))

    def linear(self, name, mod, x):
        if not self.on or x.dim() != 3:
            return
        x = x.detach().float()[0]                               # (L, C)
        prev = self.prev.get(name)
        if prev is None or prev.shape != x.shape:
            prev = torch.zeros_like(x)
        if is_spike(x):
            full, diff, e = x.abs().sum(-1), (x - prev).abs().sum(-1), E_SOP
        else:
            full, diff, e = torch.full(x.shape[:1], x.shape[1], device=x.device), (x != prev).sum(-1).float(), E_MAC
        self.prev[name] = x
        for mode, t in (("dense", full), ("delta", diff)):
            t = (t * mod.out_features * e).cpu().numpy()
            if stream_of(name) == "world":
                self.tot[mode][0] += t[:N_ROBOT].sum()
                self.tot[mode][1] += t[N_ROBOT:self.n].sum()
            else:
                self.tot[mode][2] += t.sum()


class ActionAttention:
    def __init__(self, model, horizon):
        self.on, self.rows, self.n, self.h = False, [], 0, horizon
        for layer in model.layers:
            orig = layer.attend

            def attend(q, k, v, bias, layer=layer, orig=orig, **kw):
                if self.on and kw.get("select"):
                    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias
                    a = shiftmax(scores, layer.bits)[:, :, -self.h:]          # (B, H, actions, keys)
                    w = a[..., :self.n].mean((1, 2))                           # (B, nodes)
                    self.rows.append(torch.stack([w[:, :N_ROBOT].sum(-1), w[:, N_ROBOT:].sum(-1),
                                                  a[..., -self.h:].sum(-1).mean((1, 2))], -1).cpu().numpy())
                return orig(q, k, v, bias, **kw)
            layer.attend = attend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", default=os.path.join(os.environ.get("SW_DATA", "data"), "scene_frames"))
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_RUNS", "runs"), "scene_probe_skeleton"))
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    dev = "cuda"
    pol = SkeletonPolicy(a.ckpt, k=1, flow_steps=1, fixed_noise=True, seed=a.seed)
    model, norm, cfg = pol.model, pol.norm, pol.model.cfg
    events, attn = TokenEvents(model), ActionAttention(model, cfg.horizon)
    rows = {"actions": [], "same_actions": [], "iv_actions": [], "attn": []}
    contact = []
    meta = []
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    class Ext:     # what SkeletonPolicy.set_scene reads from an extractor
        pass

    def sample(skels):
        cur = torch.cat([pol.encode(s) for s in skels])
        b = len(skels)
        desc = [d.expand(b, -1) for d in pol.desc]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.sample_actions(cur, *desc, pol.node_mask.expand(b, -1), pol.text_tokens.expand(b, -1, -1),
                                     pol.text_mask.expand(b, -1), k=1, noise=pol.noise.expand(b, -1, -1))
        return np.stack([norm.act_inv(c.float().cpu().numpy()) for c in x])

    def future(skels, noise):
        """Futures (B, H_s, n, 3) in metres, robot frame, by the world branch (10 Euler steps)."""
        cur = torch.cat([pol.encode(s) for s in skels])
        b, n = len(skels), pol.n_nodes
        desc = [d.expand(b, -1) for d in pol.desc]
        mask = pol.node_mask.expand(b, -1)
        x = noise.expand(b, -1, -1, -1).clone() * mask[:, None, :, None]
        act = torch.randn(b, cfg.horizon, cfg.action_dim, device=dev, generator=gen)
        steps = 10
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(steps):
                tau = torch.full((b,), 1 - i / steps, device=dev)
                _, v = model(cur, *desc, mask, pol.text_tokens.expand(b, -1, -1), pol.text_mask.expand(b, -1),
                             act, torch.ones(b, device=dev), x, tau)
                x = x - v.float() / steps * mask[:, None, :, None]
        x = x[:, :, :n].float().cpu().numpy()
        return np.stack([[norm.pos_inv(f) for f in xb] for xb in x])

    for path in sorted(glob.glob(os.path.join(a.frames, "*.npz"))):
        z = np.load(path)
        ext = Ext()
        ext.n_nodes, ext.node_id, ext.role, ext.slot = len(z["node_id"]), z["node_id"], z["role"], z["slot"]
        pol.set_scene(ext, str(z["language"]))
        n = ext.n_nodes
        events.n, attn.n = n, n
        fut_noise = torch.randn(1, cfg.n_future, cfg.n_max, 3, device=dev, generator=gen)
        centers = [i for i in range(N_ROBOT, n) if z["role"][i] == ROLES["center"]]
        events.prev, events.on = {}, True
        for f in range(len(z["t"])):
            sample([z["skel"][f]])
        events.on = False
        for f in range(len(z["t"])):
            ivs = np.flatnonzero(z["iv_frame"] == f)
            attn.on, attn.rows = True, []
            acts = sample([z["skel"][f]] * 2 + [z["iv_skel"][i] for i in ivs])
            attn.on = False
            w = np.mean(attn.rows, 0)                                  # (B, 3) over layers
            rows["actions"].append(acts[0][:12])
            rows["same_actions"].append(acts[1][:12])
            rows["iv_actions"] += [x[:12] for x in acts[2:]]
            rows["attn"].append(w[0])
            meta.append({"task": os.path.basename(path)[:-4], "frame": f, "t": int(z["t"][f]),
                         "closed": bool(z["closed"][f]),
                         "iv": [{"kind": str(z["iv_kind"][i]), "object": str(z["iv_object"][i]),
                                 "skel_changed": bool(np.abs(z["iv_skel"][i] - z["skel"][f]).max() > 0)}
                                for i in ivs]})
            # world-model contact test
            skel = z["skel"][f]
            d = {c: np.linalg.norm(skel[c] - skel[GRIP]) for c in centers}
            held = [c for c in centers if z["closed"][f] and d[c] < 0.06]
            if held:
                c = held[0]
                obj = np.flatnonzero(z["slot"] == z["slot"][c])
                opened = skel.copy()
                axis = skel[TIPS[0]] - skel[TIPS[1]]
                axis /= max(np.linalg.norm(axis), 1e-6)
                opened[TIPS[0]] += 0.02 * axis
                opened[TIPS[1]] -= 0.02 * axis
                away = skel.copy()
                hdir = skel[c, :2] - skel[GRIP, :2]
                hdir = hdir / np.linalg.norm(hdir) if np.linalg.norm(hdir) > 1e-3 else np.array([1.0, 0.0])
                away[obj, :2] += 0.10 * hdir
                futs = future([skel, opened, away], fut_noise)
                contact.append({"task": meta[-1]["task"], "frame": f, "kind": "held", "obj": obj.tolist(),
                                "cur": [skel.tolist(), opened.tolist(), away.tolist()], "fut": futs.tolist()})
            elif centers and min(d.values()) > 0.15:
                futs = future([skel], fut_noise)
                contact.append({"task": meta[-1]["task"], "frame": f, "kind": "free",
                                "obj": [np.flatnonzero(z["slot"] == z["slot"][c]).tolist() for c in centers],
                                "cur": [skel.tolist()], "fut": futs.tolist()})
        print(path, len(meta), flush=True)

    def follow(cur, fut, obj):
        g = fut[:, GRIP, :2] - cur[GRIP, :2]
        o = fut[:, obj, :2].mean(1) - cur[obj, :2].mean(0)
        ok = np.linalg.norm(g, axis=-1) >= 0.01
        return [float(o[i] @ g[i] / (g[i] @ g[i])) for i in np.flatnonzero(ok)], \
            float(np.linalg.norm(o, axis=-1).mean())

    res = {"frames": len(meta), "meta": meta}
    np.savez_compressed(a.out + ".npz", **{k: np.stack(v) for k, v in rows.items()})
    tot = {m: (t / len(meta)) * 1e6 for m, t in events.tot.items()}
    res["events_uJ_per_frame"] = {m: dict(zip(GROUPS, t.tolist())) for m, t in tot.items()}
    summary = {}
    for kind, names in (("held", ("held", "open", "away")), ("free", ("free",))):
        for j, nm in enumerate(names):
            ratios, disp = [], []
            for r in contact:
                if r["kind"] != kind:
                    continue
                cur, fut = np.array(r["cur"][j]), np.array(r["fut"][j])
                objs = [r["obj"]] if kind == "held" else r["obj"]
                for ob in objs:
                    rr, dd = follow(cur, fut, ob)
                    ratios += rr
                    disp.append(dd)
            if disp:
                summary[nm] = {"follow_ratio_mean": float(np.mean(ratios)) if ratios else None,
                               "follow_ratio_median": float(np.median(ratios)) if ratios else None,
                               "n_offsets": len(ratios), "object_xy_displacement_cm": 100 * float(np.mean(disp)),
                               "n_frames": len(disp)}
    res["contact"] = summary
    with open(a.out + ".json", "w") as f:
        json.dump(res, f)
    with open(a.out + "_contact.json", "w") as f:
        json.dump(contact, f)
    print(json.dumps({k: res[k] for k in ("events_uJ_per_frame", "contact")}, indent=1))


if __name__ == "__main__":
    main()
