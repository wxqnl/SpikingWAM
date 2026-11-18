"""Why a front end trained on clean LIBERO fails on LIBERO-Plus: no training, one frame per task.

Part 1, on real LIBERO-Plus scenes (first frame after settling): per category, the error of each view's own estimate, of the
confidence-weighted fusion, the pixel error, and the weight given to the third-person view. Overlays: true nodes green, the
third-person estimate red.
Part 2, on held-out clean frames: (a) the image is translated and the gain of the predicted pixel is measured (1 = the output
follows the image, 0 = the output ignores it); (b) the error of predicting each node's per-task mean position.

  source setup/env.sh; source setup/libero_plus_env.sh; MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=7 \
    python scripts/perception_why_plus.py --ckpt runs/perception/s160/ckpt.pt --tag s160
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def uv_of(model, out, v):
    return torch.stack([(out["logit_u"][v].softmax(-1) * model.uv_bins).sum(-1), (out["logit_v"][v].softmax(-1) * model.uv_bins).sum(-1)], -1)


def run_chunk(args):
    chunk, a, instructions = args
    torch.set_num_threads(1)
    from libero.libero.envs.env_wrapper import ControlEnv
    from spikewam.eval_libero_plus import DUMMY, bounded_reset, corrupt_agentview
    from spikewam.libero_tasks import NUM_STEPS_WAIT, bddl_path, get_suite, make_env, scene_objects
    from spikewam.perception import GRIP, IMAGE, N_ROBOT, TIPS, FrontEnd, project
    from spikewam.skeleton import SkeletonExtractor
    from spikewam.text import TextEncoder
    ControlEnv.reset = bounded_reset
    device = torch.device("cuda")
    front = FrontEnd(a.ckpt, device)
    enc = TextEncoder(device, pool="mean", encoder="spikeminilm16")
    rows = []
    for suite_name, idx, category, save in chunk:
        suite = get_suite(suite_name)
        task = suite.get_task(idx)
        try:
            env = make_env(bddl_path(task), has_offscreen_renderer=True)
            noise, env.noise = env.noise, 0
            front.corrupt = (lambda im, n=noise: corrupt_agentview(im, n)) if noise else None
            env.seed(7)
            env.reset()
            env.set_init_state(suite.get_task_init_states(idx)[0])
            for _ in range(NUM_STEPS_WAIT):
                env.step(DUMMY)
            names, ooi = scene_objects(env)
            ext = SkeletonExtractor(env.sim, names, ooi)
            skel = ext(env.sim)
            images, mats = front.views(env.sim, ext)
            x = torch.as_tensor(images, device=device).float().permute(0, 3, 1, 2) / 255
            x = F.interpolate(x, size=(front.cfg.size, front.cfg.size), mode="bilinear", antialias=True)
            ids = torch.as_tensor(np.asarray(ext.node_id[N_ROBOT:], np.int64), device=device)[None]
            roles = torch.as_tensor(np.asarray(ext.role[N_ROBOT:], np.int64), device=device)[None]
            grip = torch.as_tensor(np.r_[skel[GRIP], np.linalg.norm(skel[TIPS[0]] - skel[TIPS[1]])], dtype=torch.float32, device=device)[None]
            text = enc([instructions[task.name]])[0][:, 0].float()
            M = torch.as_tensor(mats, device=device)[None]
            kind = None
            if front.cfg.kinds:
                index = {k: i for i, k in enumerate(front.cfg.kinds)}
                kind = torch.as_tensor([index.get(k, 0) for k in ext.kinds], device=device)[None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = front.model(((x - 0.5) / 0.25)[None], M, text, ids, roles, grip, kind)
            true = torch.as_tensor(skel[N_ROBOT:], dtype=torch.float32, device=device)[None]
            conf = torch.stack([c.float() for c in out["conf"]], 0).softmax(0)[0, 0]
            rec = {"category": category, "name": task.name, "fused": ((out["fused"].float() - true).norm(dim=-1)[0] * 100).cpu().tolist(), "agent_weight": conf.cpu().tolist()}
            for v, cam in enumerate(("agent", "wrist")):
                uv, d = project(M[:, v].float(), true)
                vis = (d > 0) & (uv >= 0).all(-1) & (uv < IMAGE).all(-1)
                pred = uv_of(front.model, out, v).float()
                rec[cam + "_vis"] = vis[0].cpu().tolist()
                rec[cam + "_cm"] = ((out["xyz"][v].float() - true).norm(dim=-1)[0] * 100).cpu().tolist()
                rec[cam + "_px"] = (pred - uv).norm(dim=-1)[0].cpu().tolist()
                if save and v == 0:
                    im = images[0].copy()
                    for (a0, b0), (c0, d0) in zip(uv[0].cpu().numpy().astype(int), pred[0].cpu().numpy().astype(int)):
                        if 2 <= a0 < 254 and 2 <= b0 < 254:
                            im[b0 - 2:b0 + 2, a0 - 2:a0 + 2] = (0, 255, 0)
                        if 2 <= c0 < 254 and 2 <= d0 < 254:
                            im[d0 - 2:d0 + 2, c0 - 2:c0 + 2] = (255, 0, 0)
                    rec["overlay"] = np.r_[im, images[1]].tolist()
            rows.append(rec)
            env.close()
        except Exception as exc:
            rows.append({"category": category, "name": task.name, "error": f"{type(exc).__name__}: {exc}"[:200]})
    return rows


@torch.no_grad()
def clean_checks(a):
    import importlib.util
    from spikewam.perception import IMAGE, FrontEnd, N_ROBOT, project
    spec = importlib.util.spec_from_file_location("tp", "scripts/train_perception.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    device = torch.device("cuda")
    front = FrontEnd(a.ckpt, device)
    data = tp.Data(os.path.join(os.environ["SW_DATA"], "perception_rs"), device, "spikeminilm16")
    idx = data.val_idx[np.linspace(0, len(data.val_idx) - 1, 1500).astype(int)]
    res = {}
    for name, (dx, dy) in {"right_24px": (24, 0), "down_24px": (0, 24)}.items():
        gains, gains_w = [], []
        for k, images in data.loader(idx, 64, 4):
            i = torch.as_tensor(idx[k.numpy()], device=device)
            x = images.to(device).float().permute(0, 1, 4, 2, 3) / 255
            b = x.shape[0]
            run = lambda y: front.model(((F.interpolate(y.flatten(0, 1), size=(front.cfg.size, front.cfg.size), mode="bilinear", antialias=True) - 0.5) / 0.25).unflatten(0, (b, 2)), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o0 = run(x)
                for v, store in ((0, gains), (1, gains_w)):
                    y = x.clone()
                    y[:, v] = torch.roll(x[:, v], (dy, dx), (-2, -1))
                    o1 = run(y)
                    uv, d = project(data.M[i][:, v], data.skel[i])
                    vis = data.mask[i] & (d > 0) & (uv >= 40).all(-1) & (uv < IMAGE - 40).all(-1)
                    delta = (uv_of(front.model, o1, v) - uv_of(front.model, o0, v)).float()
                    store.append((delta[..., 0 if dx else 1][vis] / (dx or dy)).cpu())
        res[name] = {"agent_gain": float(torch.cat(gains).mean()), "wrist_gain": float(torch.cat(gains_w).mean())}
    # the error of guessing each node's mean position of its task, on held-out frames
    errs, net = [], []
    train_mask = torch.zeros(len(data.task), dtype=torch.bool, device=device)
    train_mask[torch.as_tensor(data.train_idx, device=device)] = True
    for t in range(len(data.names)):
        tr = (data.task == t) & train_mask
        va = (data.task == t) & ~train_mask
        m = data.skel[tr].mean(0, keepdim=True)
        e = (data.skel[va] - m).norm(dim=-1)[data.mask[va]]
        errs.append(e)
    e = torch.cat(errs) * 100
    res["task_mean_guess_cm"] = {"p50": float(e.median()), "p90": float(e.quantile(0.9)), "mean": float(e.mean())}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--per_category", type=int, default=24)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--worst", default="", help="episodes.jsonl: instead of a sample, the tasks with the largest closed-loop node error, per category")
    p.add_argument("--n_worst", type=int, default=12)
    p.add_argument("--no_clean", action="store_true", help="skip the checks on clean frames")
    a = p.parse_args()
    from libero.libero import get_libero_path
    out = os.path.join("runs", "perception", "checks", f"why_plus_{a.tag}")
    os.makedirs(out, exist_ok=True)
    table = json.load(open(os.path.join(get_libero_path("benchmark_root"), "benchmark", "task_classification.json")))
    instructions = json.load(open(os.path.join(os.environ["SW_BENCH"], "libero_plus_instructions.json")))["instructions"]
    jobs = [(s, row["id"] - 1, row["category"]) for s in table for row in table[s]]
    pick = []
    for c in sorted({j[2] for j in jobs}):
        rows = [j for j in jobs if j[2] == c]
        step = len(rows) // a.per_category
        pick += [(*r, k < 3) for k, r in enumerate(rows[::step][:a.per_category])]
    if a.worst:
        recs = [json.loads(l) for l in open(a.worst)]
        pick = []
        for c in sorted({r["category"] for r in recs}):
            rows = sorted([r for r in recs if r["category"] == c], key=lambda r: -r["percept_cm"])[:a.n_worst]
            pick += [(r["suite"], r["task_index"], c, True) for r in rows]
    chunks = [(pick[i:i + 6], a, instructions) for i in range(0, len(pick), 6)]
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_chunk, chunks):
            rows += r
            print(len(rows), "/", len(pick), flush=True)
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    summary = {"n_failed": len(bad), "failed": bad[:3]}
    for c in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == c]
        cat = lambda k: np.concatenate([np.asarray(r[k], float) for r in rs])
        s = {"fused_cm_p50": float(np.median(cat("fused"))), "agent_weight": float(cat("agent_weight").mean())}
        for cam in ("agent", "wrist"):
            vis = cat(cam + "_vis").astype(bool)
            s[cam + "_seen"] = float(vis.mean())
            s[cam + "_cm_p50"] = float(np.median(cat(cam + "_cm")[vis])) if vis.any() else None
            s[cam + "_px_p50"] = float(np.median(cat(cam + "_px")[vis])) if vis.any() else None
        summary[c] = s
        panels = [np.asarray(r["overlay"], np.uint8) for r in rs if "overlay" in r][:12]
        if panels:
            rows4 = [np.concatenate(panels[i:i + 6], 1) for i in range(0, len(panels), 6)]
            rows4 = [np.pad(r, ((0, 0), (0, rows4[0].shape[1] - r.shape[1]), (0, 0))) for r in rows4]
            Image.fromarray(np.concatenate(rows4, 0)).save(os.path.join(out, c.replace(" ", "_") + ".png"))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    if not a.no_clean:
        summary["clean_checks"] = clean_checks(a)
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
