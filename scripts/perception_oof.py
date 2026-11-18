"""Out-of-fold front-end estimates for retraining the policy on what it will see.

Two front ends (runs/perception/fold0, fold1) were each trained without half of the demonstrations. Each estimates the object
nodes of the demonstrations it never saw, on the rendered frames of data/perception_rs. The estimation error is added to the
stored skeletons of those demonstrations: for every frame `skel_in` = true skeleton + error of the nearest rendered frame.
Arm nodes stay exact (encoders and forward kinematics). spikewam/data.py feeds `skel_in` as the current skeleton; actions and
future skeletons stay true.

  CUDA_VISIBLE_DEVICES=7 python scripts/perception_oof.py --out data/skeletons_est
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from spikewam.perception import GRIP, N_ROBOT, TIPS, PerceptConfig, SpikePercept
from spikewam.text import TextEncoder


def load(path, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(device).eval()
    model.load_state_dict(st["model"])
    return model, cfg


@torch.no_grad()
def estimate(model, cfg, rgb, z, idx, text, device, batch=256):
    """Object nodes (len(idx), n, 3) estimated from the frames idx of one task."""
    s, n = z["skel"], z["skel"].shape[1] - N_ROBOT
    grip = torch.as_tensor(np.c_[s[:, GRIP], np.linalg.norm(s[:, TIPS[0]] - s[:, TIPS[1]], axis=-1)], dtype=torch.float32, device=device)
    M = torch.as_tensor(z["M"], dtype=torch.float32, device=device)
    ids, roles = [torch.as_tensor(z[k][N_ROBOT:], device=device)[None] for k in ("node_id", "role")]
    out = []
    for i in range(0, len(idx), batch):
        k = idx[i:i + batch]
        x = torch.as_tensor(np.array(rgb[k]), device=device).float().permute(0, 1, 4, 2, 3) / 255
        b = x.shape[0]
        x = F.interpolate(x.flatten(0, 1), size=(cfg.size, cfg.size), mode="bilinear", antialias=True)
        x = ((x - 0.5) / 0.25).unflatten(0, (b, 2))
        t = torch.as_tensor(k, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x, M[t], text.expand(b, -1), ids.expand(b, -1), roles.expand(b, -1), grip[t])
        out.append(o["fused"].float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rs"))
    p.add_argument("--skeletons", default=os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"))
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_DATA", "data"), "skeletons_est"))
    p.add_argument("--fold0", default="runs/perception/fold0/ckpt.pt")
    p.add_argument("--fold1", default="runs/perception/fold1/ckpt.pt")
    a = p.parse_args()
    device = torch.device("cuda")
    models = [load(a.fold0, device), load(a.fold1, device)]
    enc = TextEncoder(device, pool="mean", encoder="spikeminilm16")
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, "*")) if os.path.exists(os.path.join(d, "meta.npz")))
    all_err, by_task = [], {}
    for d in dirs:
        name = os.path.basename(d)
        suite, tid = name.rsplit("_", 1)
        src = glob.glob(os.path.join(a.skeletons, suite, f"{tid}*.npz"))
        assert len(src) == 1, (name, src)
        z = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=True))
        rgb = np.load(os.path.join(d, "rgb.npy"), mmap_mode="r")
        text = enc([str(z["language"])])[0][:, 0].float()
        ep, t = z["ep"], z["t"]
        n_obj = z["skel"].shape[1] - N_ROBOT
        err = np.zeros((len(ep), n_obj, 3), np.float32)
        for k, (model, cfg) in enumerate(models):
            idx = np.flatnonzero(ep % 2 == k)                  # fold k was trained without these demonstrations
            err[idx] = estimate(model, cfg, rgb, z, idx, text, device) - z["skel"][idx, N_ROBOT:]
        f = dict(np.load(src[0], allow_pickle=True))
        skel_in = f["skel"].copy()
        check = []
        for e, (st, ln) in enumerate(zip(f["ep_start"], f["ep_len"])):
            mine = np.flatnonzero(ep == e)
            frames = t[mine]
            assert frames.min() >= 1 and frames.max() < ln, (name, e, frames.max(), ln)
            near = np.abs(np.arange(ln)[:, None] - frames[None]).argmin(1)     # the nearest rendered frame of every step
            skel_in[st:st + ln, N_ROBOT:] += err[mine[near]]
            check.append(np.abs(f["skel"][st + frames, :N_ROBOT] - z["skel"][mine, :N_ROBOT]).max())
        assert max(check) < 1e-3, (name, max(check))             # arm nodes of the two files agree, so the frames line up
        os.makedirs(os.path.join(a.out, suite), exist_ok=True)
        np.savez(os.path.join(a.out, suite, os.path.basename(src[0])), **f, skel_in=skel_in.astype(np.float32))
        e = np.linalg.norm(err, axis=-1).ravel() * 100
        all_err.append(e)
        by_task[name] = float(np.median(e))
        print(f"{name} nodes {n_obj} median {np.median(e):.2f} cm p90 {np.percentile(e, 90):.2f} cm", flush=True)
    e = np.concatenate(all_err)
    res = {"nodes": len(e), "mean": float(e.mean()), "p50": float(np.median(e)), "p90": float(np.percentile(e, 90)), "within_1cm": float((e < 1).mean()), "within_2cm": float((e < 2).mean()), "by_task_median": by_task}
    print({k: v for k, v in res.items() if k != "by_task_median"}, flush=True)
    json.dump(res, open(os.path.join(a.out, "oof_error.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
