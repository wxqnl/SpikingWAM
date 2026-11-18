"""Action response of SpikingWAM with the perception front end: the scene_probe_skeleton.py probe, but the object nodes are
estimated from the rendered images of each frame and of each intervention, as in the closed-loop evaluation.

Inputs: the scene_frames.py renders (both cameras, every intervention rendered again) and the camera matrices and object
kinds of scene_frames_cams.py. The front end runs as deployed (spikewam.perception.FrontEnd.__call__, its renderer
replaced by the saved images; default decoding, no smoothing, no hold, so one call per plan with no state). The images
are flipped from the policy view of scene_frames.py back to the front end's upright view, as perception_regions.py does.
The policy and its settings are those of scene_probe_skeleton.py (the deployed LIBERO policy: EMA weights, k = 1,
1 flow step, one starting noise per task reused at every frame); the frame and its interventions go in one batch,
the frame twice (null intervention).

Outputs, read by scene_probe_summary.py like the skeleton probe:
  <out>.json      meta per frame (as scene_probe_skeleton.py) and front-end statistics:
                  error of the estimated object nodes to the simulator's (cm) on the frames, and the mean shift of the
                  estimated object nodes caused by each kind of intervention (cm), next to the true shift;
  <out>.npz       actions, same_actions, iv_actions (first 12 steps), attn;
  <out>_est.npz   the estimated skeletons: <task>/skel (frames) and <task>/iv_skel (interventions).

  source setup/env.sh; CUDA_VISIBLE_DEVICES=6 $SW_PY scripts/scene_probe_frontend.py \
      --ckpt runs/f2tm16_geo_finecur16_textmean/ckpt_60000.pt --perception runs/perception/dr160XL/ckpt.pt
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_probe_skeleton import ActionAttention  # noqa: E402
from spikewam.perception import N_ROBOT, FrontEnd  # noqa: E402
from spikewam.policy import SkeletonPolicy  # noqa: E402

KINDS = ("task", "other", "light", "tint")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--perception", required=True)
    p.add_argument("--frames", default=os.path.join(os.environ.get("SW_DATA", "data"), "scene_frames"))
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = a.out or os.path.join(os.environ.get("SW_RUNS", "runs"),
                                "scene_probe_frontend_" + os.path.basename(os.path.dirname(a.perception)))
    dev = "cuda"
    pol = SkeletonPolicy(a.ckpt, k=1, flow_steps=1, fixed_noise=True, seed=a.seed)
    model, norm, cfg = pol.model, pol.norm, pol.model.cfg
    attn = ActionAttention(model, cfg.horizon)
    fe = FrontEnd(a.perception, dev)
    assert not fe.smooth and not fe.hold
    rows = {"actions": [], "same_actions": [], "iv_actions": [], "attn": []}
    meta, est_out = [], {}
    err_cm, shift = [], {k: {"est": [], "true": []} for k in KINDS}

    class Ext:
        pass

    def sample(skels):
        cur = torch.cat([pol.encode(s) for s in skels])
        b = len(skels)
        desc = [d.expand(b, -1) for d in pol.desc]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.sample_actions(cur, *desc, pol.node_mask.expand(b, -1), pol.text_tokens.expand(b, -1, -1),
                                     pol.text_mask.expand(b, -1), k=1, noise=pol.noise.expand(b, -1, -1))
        return np.stack([norm.act_inv(c.float().cpu().numpy()) for c in x])

    def estimate(ext, rgb, M, skel):
        """rgb (2, H, W, 3) in the policy view of scene_frames.py; the front end's call with these images."""
        images = np.ascontiguousarray(rgb[:, :, ::-1])
        fe.views = lambda sim, e: (images, M)
        return fe(None, ext, skel, pol.text_tokens)

    for path in sorted(glob.glob(os.path.join(a.frames, "*.npz"))):
        if path.endswith("_cams.npz"):
            continue
        z, c = np.load(path), np.load(path[:-4] + "_cams.npz")
        task = os.path.basename(path)[:-4]
        ext = Ext()
        ext.n_nodes, ext.node_id, ext.role, ext.slot = len(z["node_id"]), z["node_id"], z["role"], z["slot"]
        ext.kinds = [str(k) for k in c["kinds"]]
        assert len(ext.kinds) == ext.n_nodes - N_ROBOT, task
        pol.set_scene(ext, str(z["language"]))
        attn.n = ext.n_nodes
        est_f, est_iv = [], []
        for f in range(len(z["t"])):
            M = c["M"][f]
            base = estimate(ext, z["rgb"][f], M, z["skel"][f])
            err_cm.append(np.linalg.norm(base[N_ROBOT:] - z["skel"][f][N_ROBOT:], axis=-1) * 100)
            ivs = np.flatnonzero(z["iv_frame"] == f)
            ests = [estimate(ext, z["iv_rgb"][i], M, z["iv_skel"][i]) for i in ivs]
            for i, e in zip(ivs, ests):
                k = str(z["iv_kind"][i])
                shift[k]["est"].append(float(np.linalg.norm(e[N_ROBOT:] - base[N_ROBOT:], axis=-1).mean() * 100))
                shift[k]["true"].append(float(np.linalg.norm(z["iv_skel"][i][N_ROBOT:] - z["skel"][f][N_ROBOT:], axis=-1).mean() * 100))
            attn.on, attn.rows = True, []
            acts = sample([base, base] + ests)
            attn.on = False
            rows["actions"].append(acts[0][:12])
            rows["same_actions"].append(acts[1][:12])
            rows["iv_actions"] += [x[:12] for x in acts[2:]]
            rows["attn"].append(np.mean(attn.rows, 0)[0])
            meta.append({"task": task, "frame": f, "t": int(z["t"][f]), "closed": bool(z["closed"][f]),
                         "iv": [{"kind": str(z["iv_kind"][i]), "object": str(z["iv_object"][i]),
                                 "skel_changed": bool(np.abs(e - base).max() > 0)} for i, e in zip(ivs, ests)]})
            est_f.append(base)
            est_iv += ests
        est_out[task + "/skel"] = np.array(est_f, np.float32)
        est_out[task + "/iv_skel"] = np.array(est_iv, np.float32)
        print(task, len(meta), flush=True)

    err = np.concatenate(err_cm)
    res = {"ckpt": a.ckpt, "perception": a.perception, "frames": len(meta), "meta": meta,
           "front_end": {"node_error_cm": {"mean": float(err.mean()), "median": float(np.median(err)),
                                           "p90": float(np.percentile(err, 90))},
                         "object_node_shift_cm": {k: {"estimated": float(np.mean(v["est"])), "true": float(np.mean(v["true"])),
                                                      "n": len(v["est"])} for k, v in shift.items()}}}
    np.savez_compressed(out + ".npz", **{k: np.stack(v) for k, v in rows.items()})
    np.savez_compressed(out + "_est.npz", **est_out)
    with open(out + ".json", "w") as f:
        json.dump(res, f)
    print(json.dumps(res["front_end"], indent=1))


if __name__ == "__main__":
    main()
