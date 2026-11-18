"""Operations of the coincidence attention (spikewam.geometry), which spikewam.events does not count.

Same replanning steps as scripts/energy.py (first demonstration of every task, every --stride steps) and the same
rules as spikewam.events: a unit query spike adds one key entry per valid key (the AND of q and k rings), and a unit
coincidence adds its learned gain once in every layer. The skeleton does not change within a replanning step, so the
counts are per replanning step whatever the number of samples and flow steps.

  dense   every coincidence recomputed at every replanning step
  delta   change-driven: only ring bits and coincidences that changed since the previous replanning step (the mode
          of the 0.644 mJ figure); the first step of an episode starts from zero, as in spikewam.events

  python scripts/energy_coincidence.py --ckpt runs/<run>/ckpt_60000.pt
  python scripts/energy_coincidence.py --ckpt runs/rt_f2ppt16_big/ckpt_150000.pt --bench robotwin
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from spikewam.data import Normalizer
from spikewam.train import MODELS

E_MAC, E_SOP = 4.6e-12, 0.9e-12


def codes(model, cur):
    geo, pos = model.geo, model.metric(cur)
    n, heads, g = pos.shape[1], model.cfg.heads, len(model.grips)
    anchors = torch.cat([pos, pos[:, model.grips]], 1)
    rq = geo.code.rings(anchors[:, None] + geo.delta[None, :, None]).float()     # (1, H, N + G, R, n)
    rk = geo.code.rings(pos).float()                                             # (1, N, R, n)
    ov = torch.einsum("bhqrn,bkrn->bhqkr", rq, rk)                               # (1, H, N + G, N, R)
    used = torch.zeros(heads, n + g, dtype=torch.bool)                           # rows the masks actually read
    used[:, :n] = True
    used[torch.arange(heads), n + torch.arange(heads) * g // heads] = True
    return rq, rk, ov, used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--bench", default="libero", choices=["libero", "robotwin"])
    p.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean", help="RoboTwin skeletons")
    p.add_argument("--skel_noise", type=float, default=0.0, help="std (m) on object keypoints, as in scripts/energy.py")
    p.add_argument("--est", default="", help="LIBERO: front-end skeletons, as in scripts/energy.py --est")
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    model = cls(cfg_cls(**ck["cfg"])).eval()
    model.load_state_dict(ck["ema"])
    with open(os.path.join(os.path.dirname(a.ckpt), "stats.json")) as f:
        norm = Normalizer(json.load(f))
    files = sorted(glob.glob(os.path.join(a.data, "*.npz"))) if a.bench == "robotwin" else \
        sorted(glob.glob(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons", "*", "*.npz")))
    depth, code = model.cfg.depth, model.geo.code
    est = np.load(a.est) if a.est else None
    assert est is None or a.bench == "libero"
    total = {k: 0.0 for k in ("and_dense", "gain_dense", "and_delta", "gain_delta", "phase_mac", "pairs_rings")}
    n_replans = 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for path in files:
            d = np.load(path, allow_pickle=True)
            start, length = d["ep_start"][0], d["ep_len"][0]
            n = int(d["n_nodes"][0]) if a.bench == "robotwin" else d["skel"].shape[1]
            prev = None
            obj = (d["role"][0, :n] if a.bench == "robotwin" else d["role"]) != 0
            for t in range(0, int(length), a.stride):
                skel = d["skel"][start + t, :n].astype(np.float32)
                if est is not None:
                    es = est[f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)[:-4]}/skel"]
                    assert len(es) == len(range(0, int(length), a.stride)), (path, len(es))
                    assert np.abs(es[t // a.stride][~obj] - skel[~obj]).max() < 1e-4, "robot nodes differ"
                    skel = es[t // a.stride].astype(np.float32)
                if a.skel_noise:
                    skel[obj] += rng.normal(0.0, a.skel_noise, (int(obj.sum()), 3)).astype(np.float32)
                rq, rk, ov, used = codes(model, torch.as_tensor(norm.pos(skel))[None])
                q_used, ov_used = rq[:, used], ov[:, used]
                if prev is None:
                    prev = [torch.zeros_like(x) for x in (q_used, rk, ov_used)]
                total["and_dense"] += float(q_used.sum()) * n
                total["gain_dense"] += float(ov_used.sum()) * depth
                total["and_delta"] += float((q_used - prev[0]).abs().sum()) * n + \
                    float((rk - prev[1]).abs().sum()) * int(used.sum())
                total["gain_delta"] += float((ov_used - prev[2]).abs().sum()) * depth
                # projections and phases of the head-shifted anchors and of the keys (3 per direction, 1 per ring)
                total["phase_mac"] += (3 * code.n_dirs + code.n_dirs * code.n_scales) * (int(used.sum()) + n)
                total["pairs_rings"] += int(used.sum()) * n * code.n_dirs * code.n_scales * depth
                prev = [q_used, rk, ov_used]
                n_replans += 1
    per = {k: v / n_replans for k, v in total.items()}
    res = {"ckpt": a.ckpt, "bench": a.bench, "stride": a.stride, "skel_noise": a.skel_noise, "est": a.est, "n_replans": n_replans,
           "per_replan": per}
    for mode in ("dense", "delta"):
        sop = per[f"and_{mode}"] + per[f"gain_{mode}"]
        res[f"{mode}_uJ"] = sop * E_SOP * 1e6
        res[f"{mode}_with_phase_uJ"] = (sop * E_SOP + per["phase_mac"] * E_MAC) * 1e6
    res["upper_bound_all_gains_as_mac_uJ"] = (per["pairs_rings"] * E_MAC + per["and_dense"] * E_SOP) * 1e6
    print(json.dumps(res, indent=1))
    stem = f"_sn{a.skel_noise * 1000:g}mm" if a.skel_noise else ""
    stem += "_est_" + os.path.basename(a.est).replace("_est.npz", "") if a.est else ""
    with open(os.path.join(os.path.dirname(a.ckpt), f"energy_coincidence{stem}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
