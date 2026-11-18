"""RoboTwin policy energy on front-end estimates (11-15; the LIBERO counterpart is energy.py --est with
scene_probe_frontend.py). The 10.7 mJ of energy.py --bench robotwin is counted on the simulator's skeleton along the
demonstrations; here the same policy (10-step medoid chunk, k = 3, new noise at every plan) is counted along the frames of
scripts/perception_energy_rt.py (data/perception_rtcl: the plan steps of the first episode of every set, 2,150 plans),
so front end and policy are counted on the same plans.

  --mode state     object nodes from the simulator (reference on these frames)
  --mode frontend  object nodes from the front end (fused estimate, no rigid refit: the refit needs the object models of
                   the live scene, and without it the estimates move more between plans, so the count is on the high side);
                   arm nodes from the cache (forward kinematics, as in the closed loop)

  source setup/env.sh; CUDA_VISIBLE_DEVICES=6 $SW_PY scripts/rt_policy_energy_frontend.py --mode frontend \
      --perception runs/perception/rtB30tCM7/ckpt.pt --out runs/perception/rtB30tCM7/policy_energy_rtcl.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikewam.data import Normalizer  # noqa: E402
from spikewam.events import EventCounter  # noqa: E402
from spikewam.perception import PerceptConfig, SpikePercept  # noqa: E402
from spikewam.robotwin.layout import FINGERS, GRIP_NODES, N_ROBOT, SLOT_SIZE  # noqa: E402
from spikewam.text import TextEncoder  # noqa: E402
from spikewam.train import MODELS  # noqa: E402
from train_perception import model_frame_points  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/rt_f2ppt16_big/ckpt_150000.pt")
    p.add_argument("--perception", default="runs/perception/rtB30tCM7/ckpt.pt")
    p.add_argument("--cache", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rtcl"))
    p.add_argument("--mode", required=True, choices=["state", "frontend"])
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    # policy, as scripts/energy.py
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    policy = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
    policy.load_state_dict(ck["ema"])
    with open(os.path.join(os.path.dirname(a.ckpt), "stats.json")) as f:
        norm = Normalizer(json.load(f))
    ptext = TextEncoder(dev, pool=policy.cfg.text_pool, encoder=policy.cfg.text_encoder)
    counter = EventCounter(policy, persist=False)
    gen = torch.Generator(device=dev).manual_seed(0)
    # front end, inputs as scripts/perception_energy_rt.py
    st = torch.load(a.perception, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    fe = SpikePercept(cfg).to(dev).eval()
    fe.load_state_dict(st["model"])
    enc = TextEncoder(dev, pool="mean", encoder="spikeminilm16")
    enc_tok = TextEncoder(dev, pool="tokens", encoder="spikeminilm16", max_tokens=64) if getattr(cfg, "text_tokens", False) else None
    index = {k: i for i, k in enumerate(cfg.kinds)}
    g0, g1 = GRIP_NODES
    n_f = len(FINGERS)
    n_replans, errs, text_ops = 0, [], []
    for d in sorted(glob.glob(os.path.join(a.cache, "*"))):
        if not os.path.exists(os.path.join(d, "meta.npz")):
            continue
        z = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        where = np.flatnonzero((z["ep"] == 0) & (z["t"] % a.stride == 0))
        if len(where) == 0:
            continue
        rgb = np.load(os.path.join(d, "rgb.npy"), mmap_mode="r")
        n = int(z["n_nodes"][where[0]])
        instruction = str(z["language"][0][0])
        text = enc([instruction])[0][:, 0].float()
        tok, tok_mask = enc_tok([instruction]) if enc_tok is not None else (None, None)
        as_long = lambda v: torch.as_tensor(np.asarray(v[N_ROBOT:n], np.int64), device=dev)[None]
        ids, role = as_long(z["node_id"][where[0]]), as_long(z["role"][where[0]])
        vocab = [str(v) for v in z["vocab"]]
        kind = torch.as_tensor([index.get(vocab[k], 0) for k in z["kind"][where[0], N_ROBOT:n]], device=dev)[None]
        slot = (ids - N_ROBOT) // SLOT_SIZE
        shared = (kind[..., :, None] == kind[..., None, :]) & (slot[..., :, None] != slot[..., None, :])
        shared = shared.any(-1) if getattr(cfg, "dup_order", False) else torch.zeros_like(ids, dtype=torch.bool)
        desc = [torch.as_tensor(np.asarray(z[k][where[0], :n], np.int64), device=dev)[None] for k in ("node_id", "role", "slot")]
        mask = torch.ones(1, n, dtype=torch.bool, device=dev)
        tt, tm = ptext([instruction])
        if policy.cfg.text_encoder.startswith("spikeminilm"):
            text_ops.append(ptext.model.ops(**ptext.tok([instruction], return_tensors="pt").to(dev)))
        counter.new_episode(n)
        for i in where:
            skel = z["skel"][i, :n].astype(np.float32)
            if a.mode == "frontend":
                geo = None
                if getattr(cfg, "geo", False):
                    s = torch.as_tensor(skel[N_ROBOT:n], device=dev)[None]
                    geo = model_frame_points(s, ids, role, torch.ones_like(ids, dtype=torch.bool))
                    geo[shared] = 0.0
                width = lambda g: np.linalg.norm(skel[g + 1] - skel[g + n_f])
                grip = torch.as_tensor(np.r_[skel[g0], width(g0), skel[g1], width(g1)], dtype=torch.float32, device=dev)[None]
                x = torch.as_tensor(np.array(rgb[i]), device=dev).float().permute(0, 3, 1, 2) / 255
                x = F.interpolate(x, size=(cfg.size, cfg.size), mode="bilinear", antialias=True)
                M = torch.as_tensor(z["M"][i], device=dev)[None]
                with torch.no_grad():
                    est = fe(((x - 0.5) / 0.25)[None], M, text, ids, role, grip, kind, geo, tok, tok_mask)["fused"][0].float().cpu().numpy()
                errs.append(float(np.linalg.norm(est - skel[N_ROBOT:n], axis=-1).mean() * 100))
                skel = skel.copy()
                skel[N_ROBOT:n] = est
            cur = torch.as_tensor(norm.pos(skel), device=dev)[None]
            for _ in range(a.k):
                counter.new_sample()
                with torch.no_grad():
                    policy.sample_actions(cur, *desc, mask, tt, tm, k=1, generator=gen, noise=None)
            n_replans += 1
        print(f"{os.path.basename(d)}: {n_replans} plans so far" + (f", front-end error {np.mean(errs):.2f} cm" if errs else ""), flush=True)
    res = {"ckpt": a.ckpt, "perception": a.perception if a.mode == "frontend" else None, "mode": a.mode, "cache": a.cache,
           "stride": a.stride, "k": a.k, "flow_steps": policy.cfg.flow_steps, "n_replans": n_replans,
           "frontend_error_cm_mean": float(np.mean(errs)) if errs else None, **counter.result(n_replans, a.stride)}
    if text_ops:
        sop, mac = np.mean(text_ops, 0)
        res["text_encoder_per_episode"] = {"sop": float(sop), "mac": float(mac), "energy_uJ": float((sop * 0.9e-12 + mac * 4.6e-12) * 1e6)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k in ("mode", "n_replans", "frontend_error_cm_mean", "delta", "dense", "cache")}, indent=1, default=float))


if __name__ == "__main__":
    main()
