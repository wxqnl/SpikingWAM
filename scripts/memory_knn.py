"""Does a memory of the episode so far carry action information beyond the current skeleton? (implementation record 4.9)

Same leave-one-episode-out kNN as scripts/motion_knn.py (k = 5, inside every LIBERO task, actions at t, t + 8 and
t + 15), on the hand-centred positions plus
  phase   the episode progress t / T (an oracle: an upper bound for what any memory of the past can add),
  events  the number of gripper toggles so far (closing or opening, from the executed actions; what an
          integrate-only neuron fed by gripper events holds),
  motion  each node's displacement over the last control step (the F3 input),
each tried with a few weights; the lowest error per variant is reported. No network is trained.

  python scripts/memory_knn.py
"""
import glob, os, sys
import numpy as np, torch

dev = torch.device("cuda")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spikewam.geometry import GRIP_NODE

OFFSETS = (0, 8, 15)


def knn_err(X, Y, ep, k=5):
    X = torch.as_tensor(X, device=dev); Y = torch.as_tensor(Y, device=dev); ep = torch.as_tensor(ep, device=dev)
    err = torch.zeros(Y.shape[1], device=dev)
    for i in range(0, len(X), 2048):
        d = torch.cdist(X[i:i + 2048], X)
        d[ep[i:i + 2048, None] == ep[None]] = float("inf")   # leave the whole episode out
        nn_ = d.topk(k, largest=False).indices
        err += ((Y[nn_].mean(1) - Y[i:i + 2048]) ** 2).sum(0)
    return (err / len(X)).cpu().numpy()


def grp(e):
    return tuple(v for o in range(len(OFFSETS)) for v in (e[7*o:7*o+3].mean(), e[7*o+3:7*o+6].mean(), e[7*o+6]))


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

        pos = skel - skel[:, GRIP_NODE:GRIP_NODE + 1]        # hand-centred, grip keeps its own position
        pos[:, GRIP_NODE] = skel[:, GRIP_NODE]
        pos = pos.reshape(len(skel), -1)
        scale = pos.std(0).mean() + 1e-8                    # one scale for every feature group
        pos = (pos - pos.mean(0)) / scale
        phase = np.concatenate([np.arange(n) / n for n in lens])[:, None]
        events = np.zeros((len(skel), 1), np.float32)
        motion = np.zeros_like(skel)
        for s, n in zip(starts, lens):
            g = act[s:s + n, -1] > 0
            toggles = np.concatenate([[0], np.cumsum(g[1:] != g[:-1])])
            events[s + 1:s + n, 0] = toggles[:-1]            # toggles executed before step t
            motion[s + 1:s + n] = skel[s + 1:s + n] - skel[s:s + n - 1]
        motion = (motion / scale).reshape(len(skel), -1)

        res = {"pos": knn_err(pos, Y, ep)}
        for name, extra, ws in (("phase", phase, (1, 3, 10)), ("events", events, (0.3, 1, 3)),
                                ("motion", motion, (10, 20, 40))):
            errs = [knn_err(np.concatenate([pos, w * extra], 1), Y, ep) for w in ws]
            res[name] = min(errs, key=lambda e: e.sum())
        errs = [knn_err(np.concatenate([pos, w * events, 20 * motion], 1), Y, ep) for w in (0.3, 1, 3)]
        res["motion+events"] = min(errs, key=lambda e: e.sum())
        rows[f"{suite}/{f[-6:-4]}"] = res

keys = list(next(iter(rows.values())).keys())
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "all"]:
    sel = [r for t, r in rows.items() if suite == "all" or t.startswith(suite)]
    print(suite)
    for k in keys:
        print(f"  {k:14s}", " ".join(f"{v:.3f}" for v in grp(np.mean([r[k] for r in sel], 0))))
if len(sys.argv) > 1:                                       # per-task rows for libero_10
    for t, r in rows.items():
        if t.startswith("libero_10"):
            print(t, "  ".join(f"{k}:{sum(grp(r[k])[3:]):.3f}" for k in keys))
