"""Offline check of the held-object rule on the held-out demonstrations, at the rate of the plans (every 16 control steps).

While the gripper is closed on an object, the object's nodes follow the gripper: the offsets of its nodes in the gripper
frame are taken from the network at the first plan with the gripper closed near the object, and the gripper frame comes
from the arm nodes (forward kinematics). No simulator truth is used by the rule; the truth only scores it.

  --stats prints the gripper width and the object-to-gripper distance of truly held objects, to set the two thresholds.

  CUDA_VISIBLE_DEVICES=7 python scripts/perception_held.py --ckpt runs/perception/s160/ckpt.pt [--stats]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_perception import Data, prepare  # noqa: E402

from spikewam.perception import FrontEnd  # noqa: E402
from spikewam.skeleton import N_ROBOT, ROLES  # noqa: E402

PLAN_EVERY = 8      # cached frames per plan: 16 control steps at a cache stride of 2


def grip_frame(s):
    """Rotation (3, 3) and origin of the gripper frame from the 11 arm nodes."""
    o = s[8]
    z = (o - s[7]) / np.linalg.norm(o - s[7])
    x = s[9] - s[10]
    x = x - (x @ z) * z
    x = x / max(np.linalg.norm(x), 1e-6)
    return np.stack([x, np.cross(z, x), z], 1), o


def episodes(data, root):
    """Per held-out episode: the global frame indices in time order, the full skeletons, slots and roles."""
    out = []
    start = 0
    for t, name in enumerate(data.names):
        z = np.load(os.path.join(root, name, "meta.npz"), allow_pickle=True)
        n = len(z["ep"])
        val = (data.task_np[start:start + n] == t) & np.isin(start + np.arange(n), data.val_idx)
        for e in np.unique(z["ep"][val]):
            idx = np.flatnonzero((z["ep"] == e) & val)
            out.append({"task": name, "global": start + idx[np.argsort(z["t"][idx])], "skel": z["skel"][idx[np.argsort(z["t"][idx])]], "slot": z["slot"][N_ROBOT:], "role": z["role"][N_ROBOT:]})
        start += n
    return out


def rule(ep, est, width_max, dist_max):
    """Estimates (T, n, 3) after the rule, and the per-frame held slot (0 = none)."""
    est = est.copy()
    slots = sorted(set(ep["slot"].tolist()))
    centre = {s: np.flatnonzero((ep["slot"] == s) & (ep["role"] == ROLES["center"])) for s in slots}
    held, offset, log = 0, None, []
    for f in range(len(est)):
        s = ep["skel"][f]
        R, o = grip_frame(s)
        closed = np.linalg.norm(s[9] - s[10]) < width_max
        if held and not closed:
            held = 0
        if not held and closed:
            d = {k: float(np.linalg.norm(est[f, c[0]] - o)) for k, c in centre.items() if len(c)}
            if d and min(d.values()) < dist_max:
                held = min(d, key=d.get)
                nodes = np.flatnonzero(ep["slot"] == held)
                offset = (nodes, (est[f, nodes] - o) @ R)
        if held:
            est[f, offset[0]] = offset[1] @ R.T + o
        log.append(held)
    return est, np.array(log)


def rule_mode(ep, est, truth, mode, width_max=0.07, dist_max=0.06, far=0.10):
    """Variants of the offset taken at the grasp: 'close' the estimate at closure, 'far' the last estimate with the object
    away from the gripper, 'truth' the true nodes at closure (upper bound of the rigid assumption)."""
    est = est.copy()
    slots = sorted(set(ep["slot"].tolist()))
    centre = {s: np.flatnonzero((ep["slot"] == s) & (ep["role"] == ROLES["center"])) for s in slots}
    held, offset, log, last_far = 0, None, [], {}
    for f in range(len(est)):
        s = ep["skel"][f]
        R, o = grip_frame(s)
        closed = np.linalg.norm(s[9] - s[10]) < width_max
        if held and not closed:
            held = 0
        if not held:
            for k, c in centre.items():
                if len(c) and np.linalg.norm(est[f, c[0]] - o) > far:
                    last_far[k] = est[f, np.flatnonzero(ep["slot"] == k)].copy()
        if not held and closed:
            d = {k: float(np.linalg.norm(est[f, c[0]] - o)) for k, c in centre.items() if len(c)}
            if d and min(d.values()) < dist_max:
                held = min(d, key=d.get)
                nodes = np.flatnonzero(ep["slot"] == held)
                src = {"close": est[f, nodes], "far": last_far.get(held, est[f, nodes]), "truth": truth[f, nodes]}[mode]
                offset = (nodes, (src - o) @ R)
        if held:
            est[f, offset[0]] = offset[1] @ R.T + o
        log.append(held)
    return est, np.array(log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rs"))
    p.add_argument("--text_encoder", default="spikeminilm16")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--diag", action="store_true", help="compare the sources of the held offset")
    p.add_argument("--width_max", default="0.06,0.07,0.075")
    p.add_argument("--dist_max", default="0.06,0.08,0.10,0.12")
    a = p.parse_args()
    dev = torch.device("cuda")
    fe = FrontEnd(a.ckpt, dev)
    data = Data(a.root, dev, a.text_encoder)
    eps = episodes(data, a.root)
    print(len(eps), "held-out episodes", flush=True)

    if a.stats:
        w_held, d_held, w_free = [], [], []
        for ep in eps:
            sk, obj = ep["skel"], ep["skel"][:, N_ROBOT:]
            for s in set(ep["slot"].tolist()):
                c = np.flatnonzero((ep["slot"] == s) & (ep["role"] == ROLES["center"]))
                if not len(c):
                    continue
                z = obj[:, c[0], 2] - obj[0, c[0], 2]
                d = np.linalg.norm(obj[:, c[0]] - sk[:, 8], axis=-1)
                on = (z > 0.02) & (d < 0.2)
                w_held += np.linalg.norm(sk[on, 9] - sk[on, 10], axis=-1).tolist()
                d_held += d[on].tolist()
            w_free += np.linalg.norm(sk[:, 9] - sk[:, 10], axis=-1).tolist()
        q = lambda x: [round(float(v), 4) for v in np.percentile(x, [0, 5, 25, 50, 75, 95, 100])]
        print("percentiles 0,5,25,50,75,95,100")
        print("gripper width, truly lifted objects", q(w_held))
        print("object centre to grip site, truly lifted objects", q(d_held))
        print("gripper width, all frames", q(w_free))
        return

    pred = []
    for ep in eps:
        idx = ep["global"]
        out = []
        for s in range(0, len(idx), 128):
            i = torch.as_tensor(idx[s:s + 128], device=dev)
            rgb = np.stack([np.array(data.rgb[data.task_np[j]][data.local[j]]) for j in idx[s:s + 128]])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = fe.model(prepare(torch.as_tensor(rgb), fe.cfg.size, dev, False), data.M[i], data.text[data.task[i]], data.ids[i], data.roles[i], data.grip[i])
            out.append(o["fused"].float().cpu().numpy())
        n = ep["skel"].shape[1] - N_ROBOT
        pred.append(np.concatenate(out)[:, :n])

    def score(fn):
        errs, lifted, moved = [], [], []
        for ep, est in zip(eps, pred):
            frames = np.arange(0, len(est), PLAN_EVERY)
            ep_p = {**ep, "skel": ep["skel"][frames]}
            truth = ep["skel"][:, N_ROBOT:]
            e = np.linalg.norm(fn(ep_p, est[frames]) - truth[frames], axis=-1)
            rise = truth[frames, :, 2] - truth[0, :, 2]
            disp = np.linalg.norm(truth - truth[0], axis=-1).max(0) > 0.02
            errs.append(e.ravel())
            lifted.append((rise > 0.02).ravel())
            moved.append(np.tile(disp, (len(frames), 1)).ravel())
        e, l, m = np.concatenate(errs), np.concatenate(lifted), np.concatenate(moved)
        q = lambda x: {"p50": round(float(np.median(x) * 100), 2), "p90": round(float(np.percentile(x, 90) * 100), 2), "mean": round(float(x.mean() * 100), 2)}
        return {"all": q(e), "lifted": q(e[l]), "moved": q(e[m]), "over_3cm": round(float((e > 0.03).mean()), 4)}

    if a.diag:
        for mode in ("none", "close", "far", "truth"):
            E, H, Ln, Wd = [], [], [], []
            for ep, est in zip(eps, pred):
                frames = np.arange(0, len(est), PLAN_EVERY)
                truth = ep["skel"][frames, N_ROBOT:]
                ep_p = {**ep, "skel": ep["skel"][frames]}
                out, log = (est[frames], np.zeros(len(frames), int)) if mode == "none" else rule_mode(ep_p, est[frames], truth, mode)
                held_mask = np.stack([ep["slot"] == h if h else np.zeros_like(ep["slot"], bool) for h in log])
                E.append(np.linalg.norm(out - truth, axis=-1).ravel()); H.append(held_mask.ravel())
                Ln.append(((truth[..., 2] - ep["skel"][0, N_ROBOT:, 2]) > 0.02).ravel())
                W = np.linalg.norm(ep["skel"][frames, 9] - ep["skel"][frames, 10], axis=-1) < 0.07
                Wd.append(np.tile(W[:, None], (1, truth.shape[1])).ravel())
            E, H, Ln, Wd = map(np.concatenate, (E, H, Ln, Wd))
            q = lambda x: f"n={len(x)} p50 {np.median(x) * 100:.2f} p90 {np.percentile(x, 90) * 100:.2f}" if len(x) else "n=0"
            print(mode, "| all", q(E), "| rule active", q(E[H]), "| lifted & closed", q(E[Ln & Wd]), "| lifted & open", q(E[Ln & ~Wd]), "| over3cm", round(float((E > 0.03).mean()), 4), flush=True)
        return
    res = {"network_only": score(lambda ep, est: est)}
    print("network_only", res["network_only"], flush=True)
    for w in a.width_max.split(","):
        for d in a.dist_max.split(","):
            r = score(lambda ep, est: rule(ep, est, float(w), float(d))[0])
            res[f"width<{w},dist<{d}"] = r
            print(w, d, r, flush=True)
    with open(os.path.join(os.path.dirname(a.ckpt), "held.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
