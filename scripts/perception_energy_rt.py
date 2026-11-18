"""Energy of the RoboTwin front end per plan, counted as scripts/perception_energy.py counts the LIBERO one (SOP for spike
input, MAC for real input, change-driven firing sends only the change since the previous plan of the episode, decoding in full).

The frames are the policy's own closed loop (data/perception_rtcl, spikewam/robotwin/perception_cl.py): the plan steps
(control step a multiple of 16) of the first episode of every set, three cameras, real camera matrices, object kinds and,
for a front end with the geometry query, the model-frame points.

  CUDA_VISIBLE_DEVICES=7 python scripts/perception_energy_rt.py --ckpt runs/perception/rtB4g/ckpt.pt
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

from perception_energy import DECODE_FIXED, E_MAC, E_SOP, Count, MatmulCount  # noqa: E402
from spikewam.perception import BINS_UV, PerceptConfig, SpikePercept  # noqa: E402
from spikewam.robotwin.layout import FINGERS, GRIP_NODES, N_ROBOT, SLOT_SIZE  # noqa: E402
from spikewam.text import TextEncoder  # noqa: E402
from train_perception import model_frame_points  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rtcl"))
    p.add_argument("--text_encoder", default="spikeminilm16")
    p.add_argument("--stride", type=int, default=16, help="control steps between plans")
    a = p.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    counter = Count(model)
    enc = TextEncoder(dev, pool="mean", encoder=a.text_encoder)
    # the token reader sees the same instruction and queries all episode long: change-driven counting charges it at the first plan only
    enc_tok = TextEncoder(dev, pool="tokens", encoder=a.text_encoder, max_tokens=64) if getattr(cfg, "text_tokens", False) else None
    index = {k: i for i, k in enumerate(cfg.kinds)}
    g0, g1 = GRIP_NODES
    n_f = len(FINGERS)
    n_plans, nodes_total, n_views = 0, 0, cfg.views
    bins_z = getattr(cfg, "bins_z", 270)
    for d in sorted(glob.glob(os.path.join(a.cache, "*"))):
        if not os.path.exists(os.path.join(d, "meta.npz")):
            continue
        z = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        where = np.flatnonzero((z["ep"] == 0) & (z["t"] % a.stride == 0))
        if len(where) == 0:
            continue
        rgb = np.load(os.path.join(d, "rgb.npy"), mmap_mode="r")
        n = int(z["n_nodes"][where[0]])
        text = enc([str(z["language"][0][0])])[0][:, 0].float()
        tok, tok_mask = enc_tok([str(z["language"][0][0])]) if enc_tok is not None else (None, None)
        as_long = lambda v: torch.as_tensor(np.asarray(v[N_ROBOT:n], np.int64), device=dev)[None]
        ids, role = as_long(z["node_id"][where[0]]), as_long(z["role"][where[0]])
        vocab = [str(v) for v in z["vocab"]]
        kind = torch.as_tensor([index.get(vocab[k], 0) for k in z["kind"][where[0], N_ROBOT:n]], device=dev)[None]
        slot = (ids - N_ROBOT) // SLOT_SIZE
        shared = (kind[..., :, None] == kind[..., None, :]) & (slot[..., :, None] != slot[..., None, :])
        shared = shared.any(-1) if getattr(cfg, "dup_order", False) else torch.zeros_like(ids, dtype=torch.bool)
        counter.prev = {}
        for i in where:
            skel = z["skel"][i]
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
            with torch.no_grad(), MatmulCount(counter):
                model(((x - 0.5) / 0.25)[None], M, text, ids, role, grip, kind, geo, tok, tok_mask)
            n_plans += 1
            nodes_total += ids.shape[1]
    decode = nodes_total * n_views * (2 * (2 * BINS_UV + bins_z) + DECODE_FIXED)
    res = {"ckpt": a.ckpt, "size": cfg.size, "views": n_views, "n_param": sum(p.numel() for p in model.parameters()), "n_plans": n_plans,
           "mean_spikes_per_neuron_by_layer": {k: float(np.mean(v)) for k, v in counter.rate.items()}}
    for m, t in counter.tot.items():
        sop, mac = t["sop"] / n_plans, (t["mac"] + decode) / n_plans
        res[m] = {"sop": sop, "mac": mac, "uJ_per_plan": (sop * E_SOP + mac * E_MAC) * 1e6}
    out = os.path.join(os.path.dirname(a.ckpt), "energy_rtcl.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "mean_spikes_per_neuron_by_layer"}, indent=1))


if __name__ == "__main__":
    main()
