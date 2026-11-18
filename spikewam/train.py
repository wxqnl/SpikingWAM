"""Train a skeleton world-action model on the LIBERO skeletons (one model for the four suites), with --bench robotwin
on the RoboTwin 2.0 skeletons (one model for all tasks), or with --bench calvin on the CALVIN segments (default split
ABC -> D: the training skeletons of scenes A, B and C).

  python -m spikewam.train --run r0_ann [--model ann] [--steps 60000] [--batch 48] [--set key=value ...]
  torchrun --nproc_per_node 4 -m spikewam.train ... --batch 512     # data parallel: --batch is the global batch

Writes $SW_RUNS/<run>/{config.json, stats.json, log.jsonl, ckpt_<step>.pt}. A checkpoint holds the model and its EMA.
Data parallel (torchrun): every process holds the whole dataset and the model, draws its own batch / world samples,
and the gradients are averaged (one all-reduce per step) before clipping; process 0 writes the run directory.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import time

import torch
import torch.distributed as dist

from .data import ROBOTWIN, SkeletonDataset
from .spike_wam import SpikeWAM, SpikeWAMConfig
from .text import TextEncoder
from .wam import SkeleWAM, WAMConfig

MODELS = {"ann": (SkeleWAM, WAMConfig), "spike": (SpikeWAM, SpikeWAMConfig)}


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def coerce(cfg, overrides):
    """Typed overrides (dataclass field types are strings under `from __future__ import annotations`)."""
    unknown = set(overrides) - {f.name for f in dataclasses.fields(cfg)}
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    typed = {}
    for f in dataclasses.fields(cfg):
        if f.name in overrides:
            v = overrides[f.name]
            t = type(getattr(cfg, f.name))
            typed[f.name] = (v in ("1", "true", "True")) if t is bool else t(v)
    return typed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", default="ann", choices=list(MODELS))
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--save_every", type=int, default=20000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bench", default="libero", choices=["libero", "robotwin", "calvin"])
    p.add_argument("--data", default="", help="default: $SW_DATA/skeletons (LIBERO), $SW_DATA/robotwin or "
                                              "$SW_DATA/calvin/task_ABC_D/training")
    p.add_argument("--set", nargs="*", action="extend", default=[], help="key=value overrides, repeatable")
    p.add_argument("--teacher", default="", help="checkpoint whose EMA action field is distilled (weight --kd)")
    p.add_argument("--kd", type=float, default=1.0)
    p.add_argument("--init", default="", help="start from this checkpoint (same config and statistics): fine-tuning")
    p.add_argument("--perturb", default="", help="sigma_deg,p,K: corrective perturbation (RoboTwin posture_rel; "
                                                 "data.py, record 4.13), e.g. 2,0.5,8")
    p.add_argument("--input_noise", default="", help="residuals of a front end (scripts/rt_residuals.py) added to the current object nodes")
    p.add_argument("--instr", default="first", choices=["first", "seen"],
                   help="RoboTwin: each sample reads its episode's first seen instruction, or one of them at random")
    a = p.parse_args()

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert a.batch % world == 0, "the global batch must split evenly over the processes"
    main_proc = rank == 0
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    run_dir = os.path.join(os.environ.get("SW_RUNS", "runs"), a.run)
    os.makedirs(run_dir, exist_ok=True)

    cls, cfg_cls = MODELS[a.model]
    cfg = cfg_cls(**coerce(cfg_cls(), parse_overrides(a.set)))
    if a.bench == "robotwin":
        from .robotwin.layout import GRIP_NODES, MAX_SLOTS, N_NODE_IDS, N_ROLES
        cfg.action_dim, cfg.motion_dims = ROBOTWIN["action_dim"], ROBOTWIN["motion_dims"]
        cfg.grip_nodes, cfg.n_node_ids, cfg.n_roles, cfg.max_slots = GRIP_NODES, N_NODE_IDS, N_ROLES, MAX_SLOTS
    elif a.bench == "calvin":
        from .calvin.layout import GRIP_NODES, MAX_SLOTS, N_NODE_IDS, N_ROLES
        cfg.grip_nodes, cfg.n_node_ids, cfg.n_roles, cfg.max_slots = GRIP_NODES, N_NODE_IDS, N_ROLES, MAX_SLOTS
    default = {"libero": "skeletons", "robotwin": "robotwin", "calvin": os.path.join("calvin", "task_ABC_D", "training")}
    data = a.data or os.path.join(os.environ.get("SW_DATA", "data"), default[a.bench])
    ds = SkeletonDataset(data, dev, horizon=cfg.horizon, n_future=cfg.n_future, bench=a.bench,
                         posture_rel=cfg.posture_rel, posture_scale=cfg.posture_scale, proprio=cfg.proprio,
                         instr=a.instr)
    cfg.n_max = ds.n_max
    ds.motion_lag = cfg.motion_lag
    if a.input_noise:
        ds.set_input_noise(a.input_noise)
    if a.perturb:
        sigma, prob, k_return = a.perturb.split(",")
        ds.set_perturb(float(sigma), float(prob), int(k_return))
    if main_proc:
        ds.save_stats(os.path.join(run_dir, "stats.json"))
    cfg.pos_mean, cfg.pos_std = tuple(ds.norm.stats["pos_mean"]), tuple(ds.norm.stats["pos_std"])
    model = cls(cfg).to(dev)
    if world > 1:                 # same initial weights everywhere
        for x in model.state_dict().values():
            dist.broadcast(x, 0)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(x.numel() for x in model.parameters())
    if a.init:
        with open(os.path.join(os.path.dirname(a.init), "stats.json")) as f:
            assert json.load(f) == ds.norm.stats, "init checkpoint trained on other normalisation statistics"
        ik = torch.load(a.init, map_location="cpu", weights_only=False)
        assert ik["cfg"] == dataclasses.asdict(cfg), "init checkpoint has another config"
        model.load_state_dict(ik["model"])
        ema.load_state_dict(ik["ema"])

    teacher = None
    if a.teacher:
        with open(os.path.join(os.path.dirname(a.teacher), "stats.json")) as f:
            assert json.load(f) == ds.norm.stats, "teacher trained on other normalisation statistics"
        tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
        t_cls, t_cfg_cls = MODELS[tk["model_name"]]
        t_cfg = t_cfg_cls(**tk["cfg"])
        t_cfg.pos_mean, t_cfg.pos_std = cfg.pos_mean, cfg.pos_std     # older configs lack them
        teacher = t_cls(t_cfg).to(dev).eval().requires_grad_(False)
        teacher.load_state_dict(tk["ema"])
        assert (teacher.cfg.text_pool, teacher.cfg.text_encoder) == (cfg.text_pool, cfg.text_encoder), \
            "teacher and student must read the same text features"

    text_enc = TextEncoder(dev, pool=cfg.text_pool, encoder=cfg.text_encoder)
    text_all, text_mask_all = text_enc(ds.languages)
    del text_enc
    if main_proc:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"model": a.model, "cfg": dataclasses.asdict(cfg), "args": vars(a), "n_params": n_params,
                       "n_steps_data": ds.n_steps, "world": world, "tasks": ds.task_names}, f, indent=1)
        print(f"{a.run}: {n_params / 1e6:.2f}M parameters, {ds.n_steps} training steps in the data, "
              f"{len(ds.languages)} instructions, {world} process(es) x batch {a.batch // world}", flush=True)
    torch.manual_seed(a.seed + rank)         # each process draws its own samples and noise

    decay = [x for n, x in model.named_parameters() if x.ndim >= 2 and "emb" not in n and "pos" not in n]
    no_decay = [x for n, x in model.named_parameters() if not (x.ndim >= 2 and "emb" not in n and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        frac = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * frac))

    log = open(os.path.join(run_dir, "log.jsonl"), "a") if main_proc else None
    params = [x for x in model.parameters() if x.requires_grad]
    t0, acc = time.time(), {}
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = ds.sample(a.batch // world)
        text, text_mask = text_all[batch["text"]], text_mask_all[batch["text"]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.loss(batch, text, text_mask, teacher, a.kd)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        if world > 1:                 # average the gradients over the processes (one flat all-reduce)
            grads = [x.grad if x.grad is not None else torch.zeros_like(x) for x in params]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= world
            for x, g in zip(params, flat.split([g.numel() for g in grads])):
                x.grad = g.view_as(x)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + v.item()
        acc["grad_norm"] = acc.get("grad_norm", 0.0) + gnorm.item()
        if (step + 1) % a.log_every == 0 and main_proc:
            rec = {k: v / a.log_every for k, v in acc.items()} | {"step": step + 1, "lr": lr_at(step),
                                                                   "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            acc = {}
            if (step + 1) % (a.log_every * 10) == 0:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
                      flush=True)
        if ((step + 1) % a.save_every == 0 or step + 1 == a.steps) and main_proc:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step + 1,
                        "model_name": a.model, "cfg": dataclasses.asdict(cfg)},
                       os.path.join(run_dir, f"ckpt_{step + 1}.pt"))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
