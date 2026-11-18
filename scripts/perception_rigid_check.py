"""How much can the rigid shape of an object clean up its estimated nodes? No training, held-out frames.

The six box nodes of an object (center, top, four sides) move as one rigid body. The true shape is fitted to the estimated nodes
(rotation and translation), which is the upper bound of any rigid clean-up, since it uses the exact shape.

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=.:$PYTHONPATH python scripts/perception_rigid_check.py --ckpt runs/perception/dr160/ckpt.pt
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

from spikewam.perception import FrontEnd
from spikewam.skeleton import N_ROBOT, SLOT_SIZE


def kabsch(src, dst, w):
    """Move src (B, n, 3) onto dst with the rotation and translation that fit the nodes with w (B, n) true."""
    w = w.float()[..., None]
    n = w.sum(1, keepdim=True).clamp_min(1)
    cs, cd = (src * w).sum(1, keepdim=True) / n, (dst * w).sum(1, keepdim=True) / n
    H = ((src - cs) * w).transpose(1, 2) @ (dst - cd)
    U, _, Vh = torch.linalg.svd(H.float())
    d = torch.sign(torch.det(U @ Vh))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = U @ D @ Vh                                  # row vectors: x -> x R
    return (src - cs) @ R + cd


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", default="perception_rs,perception_dr")
    p.add_argument("--n", type=int, default=6000)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("tp", "scripts/train_perception.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    device = torch.device("cuda")
    front = FrontEnd(a.ckpt, device)
    res = {}
    data = tp.Data(",".join(os.path.join(os.environ["SW_DATA"], r) for r in a.roots.split(",")), device, "spikeminilm16", kinds=True)      # a root without kinds takes them from the others
    assert data.kinds == front.cfg.kinds
    for root, val in zip(data.roots, data.val_by_root):
        idx = val[np.linspace(0, len(val) - 1, a.n).astype(int)]
        raw, fit, near, common = [], [], [], []
        for k, images in data.loader(idx, 256, 8):
            i = torch.as_tensor(idx[k.numpy()], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = front.model(tp.prepare(images, front.cfg.size, device, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], data.kind[i])
            est, true, mask = out["fused"].float(), data.skel[i], data.mask[i]
            slot = (data.ids[i] - N_ROBOT) // SLOT_SIZE
            box = mask & (data.roles[i] >= 1) & (data.roles[i] <= 6)
            for s in range(4):
                m = box & (slot == s)
                keep = m.sum(1) == 6
                if not keep.any():
                    continue
                e, t, w = est[keep], true[keep], m[keep]
                f = kabsch(t, e, w)
                raw.append((e - t).norm(dim=-1)[w])
                fit.append((f - t).norm(dim=-1)[w])
                near.append((t - data.grip[i][keep][:, None, :3]).norm(dim=-1)[w])
                nw = w.float()[..., None]
                common.append((((e - t) * nw).sum(1) / 6).norm(dim=-1))
        r, f, h, c = [torch.cat(x) * 100 for x in (raw, fit, near, common)]
        q = lambda x: {"p50": round(float(x.median()), 3), "p90": round(float(x.quantile(0.9)), 3), "mean": round(float(x.mean()), 3), "within_1cm": round(float((x < 1).float().mean()), 3)}
        res[root] = {"nodes": len(r), "raw": q(r), "rigid": q(f), "near_hand_10cm_raw": q(r[h < 10]), "near_hand_10cm_rigid": q(f[h < 10]), "centroid_error": q(c)}
        print(root, json.dumps(res[root]), flush=True)
    json.dump(res, open(os.path.join("runs", "perception", "checks", "rigid_" + os.path.basename(os.path.dirname(a.ckpt)) + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
