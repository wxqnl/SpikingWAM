"""Energy account of the policy along demonstration trajectories (research plan section 4.8).

For the first --episodes demonstrations of every task, the policy replans every --stride steps on the recorded
skeleton (as in closed-loop control: K samples x flow steps per replanning step, no future branch). spikewam.events
counts the operations of three ways of running the same network: dense (recompute everything), cache (recompute only
tokens whose input changed) and delta (change-driven firing; exact, same outputs). 45 nm: E_MAC = 4.6 pJ,
E_AC = 0.9 pJ.

  python scripts/energy.py --ckpt runs/<run>/ckpt_60000.pt [--episodes 1] [--stride 16]
  python scripts/energy.py --ckpt runs/rt_f2/ckpt_60000.pt --bench robotwin     # RoboTwin skeletons (4.11)
  python scripts/energy.py --ckpt runs/<calvin run>/ckpt_30000.pt --bench calvin    # CALVIN scene D (4.23)

CALVIN: the first --episodes validation segments (scene D) of each of the 34 tasks, over the annotated frames.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from spikewam.data import Normalizer
from spikewam.events import MODES, EventCounter
from spikewam.text import TextEncoder
from spikewam.train import MODELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--flow_steps", type=int, default=0, help="Euler steps per sample (0 = the checkpoint's)")
    p.add_argument("--fixed_noise", action="store_true", help="k starting noises per episode, reused at every replan")
    p.add_argument("--persist", action="store_true",
                   help="k = 1 with --fixed_noise: the action stream carries over between replanning steps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--files", type=int, default=0, help="only the first this many task files (0 = all)")
    p.add_argument("--bench", default="libero", choices=["libero", "robotwin", "calvin"])
    p.add_argument("--skel_noise", type=float, default=0.0,
                   help="std (m) of Gaussian noise on every object keypoint at every replanning step (perception error)")
    p.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean", help="RoboTwin skeletons")
    p.add_argument("--est", default="", help="LIBERO: skeletons with front-end object nodes for the first demonstration of every task "
                   "at its replanning steps (scene_probe_frontend.py <out>_est.npz, key <suite>_<task>/skel)")
    a = p.parse_args()
    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cls, cfg_cls = MODELS[ck["model_name"]]
    model = cls(cfg_cls(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["ema"])
    if a.flow_steps:
        model.cfg.flow_steps = a.flow_steps
    with open(os.path.join(os.path.dirname(a.ckpt), "stats.json")) as f:
        norm = Normalizer(json.load(f))
    text = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)
    assert not a.persist or (a.fixed_noise and a.k == 1), "--persist: one sample from a fixed noise"
    counter = EventCounter(model, persist=a.persist)
    gen = torch.Generator(device=dev).manual_seed(0)
    rng = np.random.default_rng(0)
    n_replans = 0
    text_ops = []                 # spiking text encoder, once per episode
    if a.bench == "robotwin":
        files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    elif a.bench == "calvin":
        files = [os.path.join(os.environ.get("SW_DATA", "data"), "calvin", "task_ABC_D", "validation",
                              "calvin_scene_D.npz")]
    else:
        files = sorted(glob.glob(os.path.join(os.environ.get("SW_DATA", "data"), "skeletons", "*", "*.npz")))
    files = files[:a.files] if a.files else files
    est = np.load(a.est) if a.est else None
    assert est is None or (a.bench == "libero" and a.episodes == 1), "--est: LIBERO, first demonstration only"
    for path in files:
        d = np.load(path, allow_pickle=True)
        items = list(enumerate(zip(d["ep_start"], d["ep_len"])))[:a.episodes]
        if a.bench == "calvin":     # the first segments of each task, annotated frames only
            by_task = {}
            for e, t in enumerate(d["task"]):
                by_task.setdefault(str(t), []).append(e)
            items = [(e, (d["ep_start"][e], d["n_sample"][e])) for e in sorted(x for v in by_task.values()
                                                                               for x in v[:a.episodes])]
        for e, (start, length) in items:
            # LIBERO: one scene and instruction per file; RoboTwin: per episode (n_nodes, descriptors, language);
            # CALVIN: one scene, an instruction per segment
            n = int(d["n_nodes"][e]) if a.bench == "robotwin" else d["skel"].shape[1]
            desc = [torch.as_tensor(d[k][e, :n] if a.bench == "robotwin" else d[k], device=dev)[None].long()
                    for k in ("node_id", "role", "slot")]
            mask = torch.ones(1, n, dtype=torch.bool, device=dev)
            instruction = str(d["language"][e] if a.bench != "libero" else d["language"])
            tt, tm = text([instruction])
            if model.cfg.text_encoder.startswith("spikeminilm"):
                sop, mac = text.model.ops(**text.tok([instruction], return_tensors="pt").to(dev))
                text_ops.append((sop, mac))
            counter.new_episode(n)
            noises = [torch.randn(1, model.cfg.horizon, model.cfg.action_dim, device=dev, generator=gen)
                      for _ in range(a.k)] if a.fixed_noise else [None] * a.k
            obj = (d["role"][e, :n] if a.bench == "robotwin" else d["role"]) != 0       # role 0: robot nodes
            for t in range(0, int(length), a.stride):
                skel = d["skel"][start + t, :n].astype(np.float32)
                if est is not None:
                    es = est[f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)[:-4]}/skel"]
                    assert len(es) == len(range(0, int(length), a.stride)), (path, len(es))
                    assert np.abs(es[t // a.stride][~obj] - skel[~obj]).max() < 1e-4, "robot nodes differ"
                    skel = es[t // a.stride].astype(np.float32)
                if a.skel_noise:
                    skel[obj] += rng.normal(0.0, a.skel_noise, (int(obj.sum()), 3)).astype(np.float32)
                cur = torch.as_tensor(norm.pos(skel), device=dev)[None]
                for j in range(a.k):
                    counter.new_sample()
                    with torch.no_grad():
                        model.sample_actions(cur, *desc, mask, tt, tm, k=1, generator=gen, noise=noises[j])
                n_replans += 1
        print(f"{path}: {n_replans} replanning steps so far", flush=True)
    res = {"ckpt": a.ckpt, "model": ck["model_name"], "episodes_per_task": a.episodes, "stride": a.stride,
           "k": a.k, "flow_steps": model.cfg.flow_steps, "fixed_noise": a.fixed_noise, "persist": a.persist, "skel_noise": a.skel_noise, "est": a.est, "n_replans": n_replans, **counter.result(n_replans, a.stride)}
    if text_ops:
        sop, mac = np.mean(text_ops, 0)
        res["text_encoder_per_episode"] = {"sop": sop, "mac": mac, "energy_uJ": (sop * 0.9e-12 + mac * 4.6e-12) * 1e6}
        print(f"spiking text encoder, once per episode: SOP {sop / 1e6:.1f} M  MAC {mac / 1e6:.2f} M  "
              f"{res['text_encoder_per_episode']['energy_uJ']:.1f} uJ", flush=True)
    for mode in MODES:
        r = res[mode]
        print(f"{mode:6s} SOP {r['sop'] / 1e6:9.1f} M  MAC {r['mac'] / 1e6:8.1f} M  "
              f"{r['energy_uJ_per_replan']:8.1f} uJ/replan  {r['energy_uJ_per_control_step']:6.2f} uJ/step")
        print("        " + "  ".join(f"{k} {v / 1e6:.1f}M" for k, v in sorted(r["per_replan"].items())))
    stem = os.path.splitext(os.path.basename(a.ckpt))[0] + (f"_files{a.files}" if a.files else "")
    stem += f"_{a.bench}" if a.bench != "libero" else ""
    stem += f"_k{a.k}" if a.k != 3 else ""
    stem += f"_steps{a.flow_steps}" if a.flow_steps else ""
    stem += "_persist" if a.persist else ("_fixed" if a.fixed_noise else "")
    stem += f"_sn{a.skel_noise * 1000:g}mm" if a.skel_noise else ""
    stem += "_est_" + os.path.basename(a.est).replace("_est.npz", "") if a.est else ""
    with open(os.path.join(os.path.dirname(a.ckpt), f"energy_{stem}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
