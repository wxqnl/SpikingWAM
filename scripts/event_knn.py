"""Which motion signal keeps the future-action information without the copycat? (implementation record 4.9)

Same leave-one-episode-out kNN as scripts/motion_knn.py, on the hand-centred positions plus one of
  obj_signed  each object node's displacement over the last control step (robot nodes left out),
  obj_event   one binary event per object node: it moved more than 1 mm in the last step,
  all_event   the same event for every node,
  all_signed  every node's displacement (the F3 input).
The weight of each extra input is picked on the t + 8 / t + 15 error; the error at t measures how much of the
current action the input gives away (copycat). No network is trained.

  python scripts/event_knn.py
"""
import glob, os, sys
import numpy as np, torch

dev = torch.device("cuda")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spikewam.geometry import GRIP_NODE
OFFSETS = (0, 8, 15)
def knn_err(X, Y, ep, k=5):
    X = torch.as_tensor(X, device=dev, dtype=torch.float32); Y = torch.as_tensor(Y, device=dev); ep = torch.as_tensor(ep, device=dev)
    err = torch.zeros(Y.shape[1], device=dev)
    for i in range(0, len(X), 2048):
        d = torch.cdist(X[i:i + 2048], X)
        d[ep[i:i + 2048, None] == ep[None]] = float("inf")
        nn_ = d.topk(k, largest=False).indices
        err += ((Y[nn_].mean(1) - Y[i:i + 2048]) ** 2).sum(0)
    return (err / len(X)).cpu().numpy()
def grp(e): return tuple(v for o in range(len(OFFSETS)) for v in (e[7*o:7*o+3].mean(), e[7*o+3:7*o+6].mean(), e[7*o+6]))
rows = {}
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    for f in sorted(glob.glob(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons", suite, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        skel, act = d["skel"].astype(np.float32), d["actions"].astype(np.float32)
        starts, lens = d["ep_start"], d["ep_len"]
        Y0 = (act - act.mean(0)) / (act.std(0) + 1e-6)
        ep = np.repeat(np.arange(len(starts)), lens)
        last = np.repeat(starts + lens - 1, lens)
        Y = np.concatenate([Y0[np.minimum(np.arange(len(Y0)) + o, last)] for o in OFFSETS], 1)
        pos = skel - skel[:, GRIP_NODE:GRIP_NODE + 1]; pos[:, GRIP_NODE] = skel[:, GRIP_NODE]
        pos = pos.reshape(len(skel), -1); scale = pos.std(0).mean() + 1e-8; pos = (pos - pos.mean(0)) / scale
        dv = np.zeros_like(skel)
        for s, n in zip(starts, lens):
            dv[s + 1:s + n] = skel[s + 1:s + n] - skel[s:s + n - 1]
        obj = (d["slot"] > 0).astype(np.float32)[None, :, None]
        moved = (np.linalg.norm(dv, axis=-1) > 0.001).astype(np.float32)       # one event per node: moved > 1 mm
        res = {"pos": knn_err(pos, Y, ep)}
        for name, extra, ws in (("obj_signed", (dv * obj / scale).reshape(len(skel), -1), (10, 20, 40)),
                                ("obj_event", moved * obj[..., 0], (0.3, 1, 3)),
                                ("all_event", moved, (0.3, 1, 3)),
                                ("all_signed", (dv / scale).reshape(len(skel), -1), (20,))):
            errs = [knn_err(np.concatenate([pos, w * extra], 1), Y, ep) for w in ws]
            res[name] = min(errs, key=lambda e: e[7:].sum())          # pick the weight on t+8 / t+15
        rows[f"{suite}/{f[-6:-4]}"] = res
keys = list(next(iter(rows.values())).keys())
for suite in ["libero_10", "all"]:
    sel = [r for t, r in rows.items() if suite == "all" or t.startswith(suite)]
    print(suite, "(pos/rot/grip at t | t+8 | t+15)")
    for k in keys:
        g = grp(np.mean([r[k] for r in sel], 0))
        print(f"  {k:11s}", " | ".join("/".join(f"{v:.3f}" for v in g[i:i+3]) for i in (0, 3, 6)))
