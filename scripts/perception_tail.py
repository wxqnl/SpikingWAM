"""Where the errors of the keypoint front end sit, on the held-out demonstrations.

Groups the distance between estimated and true object nodes by node role, by whether the node moves during the episode,
by its distance to the gripper, by lifting, by visibility in the wrist view, per axis and per task, and reports for each
group its share of the frames and its share of the large errors.

  CUDA_VISIBLE_DEVICES=7 python scripts/perception_tail.py --ckpt runs/perception/s160/ckpt.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_perception import Data, prepare  # noqa: E402

from spikewam.perception import IMAGE, FrontEnd, project  # noqa: E402
from spikewam.skeleton import ROLES  # noqa: E402

LARGE = 0.03


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rs"))
    p.add_argument("--text_encoder", default="spikeminilm16")
    a = p.parse_args()
    dev = torch.device("cuda")
    fe = FrontEnd(a.ckpt, dev)
    data = Data(a.root, dev, a.text_encoder)
    ep = np.concatenate([np.load(os.path.join(a.root, n, "meta.npz"), allow_pickle=True)["ep"] for n in data.names])
    idx = data.val_idx
    first = {}
    for i in idx:
        first.setdefault((int(data.task_np[i]), int(ep[i])), int(i))
    first_idx = torch.as_tensor([first[(int(data.task_np[i]), int(ep[i]))] for i in idx], device=dev)
    all_i = torch.as_tensor(idx, device=dev)
    skel, mask, role = data.skel[all_i], data.mask[all_i], data.roles[all_i]
    est, xyz_v = [], [[], []]
    for k, images in data.loader(idx, 256, 8):
        i = all_i[k.to(dev)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = fe.model(prepare(images, fe.cfg.size, dev, False), data.M[i], data.text[data.task[i]], data.ids[i], data.roles[i], data.grip[i])
        est.append(o["fused"].cpu())
        for v in range(2):
            xyz_v[v].append(o["xyz"][v].cpu())
    est = torch.cat(est).to(dev)
    err = (est - skel).norm(dim=-1)
    # displacement from the first frame of the same episode, over the whole episode
    start = data.skel[first_idx]
    disp = (skel - start).norm(dim=-1)
    key = data.task[all_i].cpu().numpy() * 1000 + ep[idx]
    moved = torch.zeros_like(mask)
    for k in np.unique(key):
        s = torch.as_tensor(np.flatnonzero(key == k), device=dev)
        moved[s] = (disp[s].max(0).values > 0.02).expand(len(s), -1)
    lifted = (skel[..., 2] - start[..., 2]) > 0.02
    grip = (skel - data.grip[all_i][:, None, :3]).norm(dim=-1)
    uv, d = project(data.M[all_i][:, 1], skel)
    in_wrist = (d > 0) & (uv >= 0).all(-1) & (uv < IMAGE).all(-1)
    err_m, n_large = err[mask], (err[mask] > LARGE).sum().item()

    def stat(sel):
        sel = sel & mask
        e = err[sel]
        if not len(e):
            return None
        return {"share_of_nodes": round(float(sel.sum() / mask.sum()), 4), "share_of_large": round(float((e > LARGE).sum() / max(n_large, 1)), 4),
                "p50_cm": round(float(e.median() * 100), 2), "p90_cm": round(float(e.quantile(0.9) * 100), 2), "large_rate": round(float((e > LARGE).float().mean()), 4)}

    res = {"ckpt": a.ckpt, "frames": len(idx), "nodes": int(mask.sum()), "all": {"p50_cm": float(err_m.median() * 100), "p90_cm": float(err_m.quantile(0.9) * 100), "mean_cm": float(err_m.mean() * 100), "large_rate": n_large / len(err_m)},
           "by_role": {n: stat(role == r) for n, r in ROLES.items() if r},
           "moves_in_episode": {"moved": stat(moved), "static": stat(~moved)},
           "lifted": {"lifted>2cm": stat(lifted), "not": stat(~lifted)},
           "grip_distance": {"<6cm": stat(grip < 0.06), "6-12cm": stat((grip >= 0.06) & (grip < 0.12)), "12-20cm": stat((grip >= 0.12) & (grip < 0.2)), ">20cm": stat(grip >= 0.2)},
           "in_wrist_view": {"in": stat(in_wrist), "out": stat(~in_wrist)},
           "moved_near_gripper": {"moved,<6cm": stat(moved & (grip < 0.06)), "moved,6-12cm": stat(moved & (grip >= 0.06) & (grip < 0.12)), "moved,>12cm": stat(moved & (grip >= 0.12))},
           "axis_abs_cm": {ax: {"p50": round(float((est - skel)[..., j][mask].abs().median() * 100), 2), "p90": round(float((est - skel)[..., j][mask].abs().quantile(0.9) * 100), 2)} for j, ax in enumerate("xyz")}}
    per_task = {}
    for t, name in enumerate(data.names):
        s = stat(data.task[all_i][:, None].expand_as(mask) == t)
        if s:
            per_task[name] = s
    res["worst_tasks_by_p90"] = sorted(((n, s["p90_cm"], s["p50_cm"], s["share_of_large"]) for n, s in per_task.items()), key=lambda x: -x[1])[:10]
    res["best_tasks_by_p90"] = sorted(((n, s["p90_cm"], s["p50_cm"]) for n, s in per_task.items()), key=lambda x: x[1])[:5]
    with open(os.path.join(os.path.dirname(a.ckpt), "tail.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
