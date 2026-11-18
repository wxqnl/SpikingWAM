"""Does node motion carry action information beyond node positions? (implementation record 4.9)

Leave-one-episode-out kNN (k = 5) inside every LIBERO task: predict the action at t, t + 8 and t + 15 (per-dim
normalised) from the hand-centred positions (plus the grip position), and from the same plus each node's
displacement over the last control step (all nodes, robot nodes only, object nodes only). No network is trained.

  python scripts/motion_knn.py
"""
import glob, os, sys
import numpy as np, torch

dev = torch.device("cuda")
def feats(skel, starts, lens, lag, w, grip_idx, which=None):
    pos = skel - skel[:, grip_idx:grip_idx + 1]           # hand-centred
    pos[:, grip_idx] = skel[:, grip_idx]                  # keep the grip's own position
    f = [pos.reshape(len(skel), -1)]
    if lag:
        prev = np.empty_like(skel)
        for s, n in zip(starts, lens):
            idx = np.arange(n) - lag
            idx[idx < 0] = 0
            prev[s:s + n] = skel[s + idx]
        dv = skel - prev
        if which is not None:
            dv = dv * which[None, :, None]
        f.append(w * dv.reshape(len(skel), -1))
    return np.concatenate(f, 1)

def knn_err(X, Y, ep, k=5):
    X = torch.as_tensor(X, device=dev); Y = torch.as_tensor(Y, device=dev); ep = torch.as_tensor(ep, device=dev)
    X = (X - X.mean(0)) / (X[:, :X.shape[1]].std(0).mean() + 1e-8)
    err = torch.zeros(Y.shape[1], device=dev)
    for i in range(0, len(X), 2048):
        d = torch.cdist(X[i:i + 2048], X)
        d[ep[i:i + 2048, None] == ep[None]] = float("inf")   # leave the whole episode out
        nn_ = d.topk(k, largest=False).indices
        err += ((Y[nn_].mean(1) - Y[i:i + 2048]) ** 2).sum(0)
    return (err / len(X)).cpu().numpy()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spikewam.geometry import GRIP_NODE
from spikewam.skeleton import ROBOT_BODIES, ROBOT_SITES, ROBOT_TIPS
N_ROBOT = len(ROBOT_BODIES) + len(ROBOT_SITES) + len(ROBOT_TIPS)
rows = {}
OFFSETS = (0, 8, 15)
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    for f in sorted(glob.glob(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons", suite, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        skel, act = d["skel"].astype(np.float32), d["actions"].astype(np.float32)
        Y0 = (act - act.mean(0)) / (act.std(0) + 1e-6)
        ep = np.repeat(np.arange(len(d["ep_start"])), d["ep_len"])
        t_in = np.concatenate([np.arange(n) for n in d["ep_len"]])
        last = np.repeat(d["ep_start"] + d["ep_len"] - 1, d["ep_len"])
        Y = np.concatenate([Y0[np.minimum(np.arange(len(Y0)) + o, last)] for o in OFFSETS], 1)
        res = {"pos": knn_err(feats(skel, d["ep_start"], d["ep_len"], 0, 0, GRIP_NODE), Y, ep)}
        robot = (d["role"] == 0) | (np.arange(skel.shape[1]) < N_ROBOT)
        for name, which in (("all", None), ("robot", robot.astype(np.float32)), ("objects", (~robot).astype(np.float32))):
            res[f"{name}"] = knn_err(feats(skel, d["ep_start"], d["ep_len"], 1, 20.0, GRIP_NODE, which), Y, ep)
        rows[f"{suite}/{f[-6:-4]}"] = res
keys = list(next(iter(rows.values())).keys())
def grp(e): return tuple(v for o in range(len(OFFSETS)) for v in (e[7*o:7*o+3].mean(), e[7*o+3:7*o+6].mean(), e[7*o+6]))
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "all"]:
    sel = [r for t, r in rows.items() if suite == "all" or t.startswith(suite)]
    print(suite, "  ".join(f"{k}: " + "/".join(f"{v:.3f}" for v in grp(np.mean([r[k] for r in sel], 0))) for k in keys))
