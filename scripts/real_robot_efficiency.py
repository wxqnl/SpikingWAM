"""Energy, speed and GPU memory of SpikingWAM as deployed on the SO-101 arm (paper section 5.4, 11-13).

Deployed configuration (metadata.json of the recorded trials): policy SpikeWAM v2w 12,000 steps, readout 5x3mac
(5 flow steps, 3 samples, medoid consensus), tier fused; cubes by the spiking front end (so101_percept, 30,000 steps)
at every request while the cube is on the table; cups by the SDT-V3 cup locator (so101_cup_sdt, 8,000 steps) once at
the episode start; spiking MiniLM sentence text. Code: the deploy bundle zhenji/so101_deploy/spikewam (read only).

--part energy   the recorded real-robot trials (real_robot/20261112_v2w_green_left_last10, every request): the
                deployed server's path is replayed on the recorded wrist frames and states (start: front end on the
                first frame; each request: cube re-detection, tracker, skeleton, policy). Counting as the paper:
                policy by spikewam.events.EventCounter (as scripts/energy.py: k samples of sample_actions(k=1), FP32,
                change-driven 'delta'); front ends as scripts/perception_energy.py (spike input -> SOP x mean spike
                count, real input -> MAC, change-driven between calls of the same episode, batch norm left out, plus
                the SimCC decoding as MAC); text by SpikeMiniLM.ops. 45 nm: SOP 0.9 pJ, MAC 4.6 pJ.
--part speed    the 18 validation demonstrations at the collector's rhythm (every 9th tick, the protocol of the 11-09
                six-model speed table in zhenji/README.md section 6): the deployed server object (fused tier, CUDA
                Graph) called in process, no HTTP. Per call: perception + tracker + policy, the server's own
                inference_ms; the first request of an episode (front end reads the scene, SDT-V3 cup) separately.
                Peak memory: torch.cuda.max_memory_reserved / allocated over start-up and all calls, and the process's
                memory in nvidia-smi.

    CUDA_VISIBLE_DEVICES=5 zhenji/so101_deploy/envs/spike/bin/python scripts/real_robot_efficiency.py --part speed
    CUDA_VISIBLE_DEVICES=5 zhenji/so101_deploy/envs/spike/bin/python scripts/real_robot_efficiency.py --part energy
Output: real_robot/efficiency/{speed,energy}.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "real_robot/efficiency"
# compile caches in this project, not in the shared deploy bundle (runtime.py only sets them when unset)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(OUT / "cache/inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(OUT / "cache/triton"))
DEPLOY = Path("/home/user01/Minko/calude/zhenji/so101_deploy")
BUNDLE = DEPLOY / "spikewam"
sys.path.insert(0, str(BUNDLE / "deploy"))
from runtime import WAMRuntime  # noqa: E402  (sets the text-encoder paths and puts BUNDLE/code on the path)

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

RUN = ROOT / "real_robot/20261112_v2w_green_left_last10"
POLICY = Path("/home/user01/Minko/calude/zhenji/spikewam/runs/so101_wam_v2w/ckpt_12000.pt")
PERCEPT = BUNDLE / "models/so101_percept/model.pt"
CUP = BUNDLE / "models/so101_cup_sdt/model.pt"
CONFIG = DEPLOY / "collector/deployments/spikewam/config.json"
VAL = DEPLOY / "checks/val_episodes"
READOUT, K, STEPS = "5x3mac", 3, 5
E_SOP, E_MAC = 0.9e-12, 4.6e-12
PERIOD = 9


# ---- front-end counting (scripts/perception_energy.py, with Conv1d for SDT-V3's MLP) ------------------------------
SDT_LEVELS = 4                   # SDT-V3 MultiSpike outputs k / 4, k = 0..4 (scripts/image_events.py)


def spike_counts(x, levels=1):
    """Unit-spike counts of a spike tensor (integer counts, or k / levels), else None."""
    x = x.float()
    if x.min() < 0:
        return None
    s = x * levels
    return s if torch.all(s == torch.round(s)) else None


class Count:
    def __init__(self, model, sdt_prefix=None):
        """sdt_prefix: modules under it (the SDT-V3 backbone) read k / 4 spikes; the others integer spike counts."""
        self.tot = {m: defaultdict(float) for m in ("dense", "delta")}
        self.prev, self.calls = {}, 0
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                levels = SDT_LEVELS if sdt_prefix and name.startswith(sdt_prefix) else 1
                m.register_forward_hook(lambda m, inp, out, name=name, levels=levels: self.layer(name, m, inp[0], out, levels))
    def layer(self, name, m, x, out, levels=1):
        if isinstance(m, nn.Conv2d):
            ops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Conv1d):
            ops = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]
        else:
            ops = x.numel() // m.in_features * m.in_features * m.out_features
        self.account(name, x, ops, spike_counts(x, levels))

    def account(self, name, x, ops, s):
        prev = self.prev.get(name)
        if prev is not None and prev.shape != x.shape:
            prev = None
        if s is not None:
            change = s if prev is None else (s - prev).abs()
            self.tot["dense"]["sop"] += ops * s.mean().item()
            self.tot["delta"]["sop"] += ops * change.mean().item()
            self.prev[name] = s
        else:
            x = x.float()
            changed = torch.ones_like(x) if prev is None else (x != prev).float()
            self.tot["dense"]["mac"] += ops
            self.tot["delta"]["mac"] += ops * changed.mean().item()
            self.prev[name] = x

    def add_mac(self, n):
        for m in self.tot:
            self.tot[m]["mac"] += n

    def uj(self, mode, per=1):
        t = self.tot[mode]
        return {"sop": t["sop"] / per, "mac": t["mac"] / per, "uJ": (t["sop"] * E_SOP + t["mac"] * E_MAC) * 1e6 / per}


class MatmulCount(TorchFunctionMode):
    def __init__(self, counter):
        super().__init__()
        self.counter, self.calls = counter, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, "__name__", "") in ("matmul", "__matmul__") and len(args) == 2:
            a, b = args
            if a.dim() >= 3 and b.dim() >= 3:     # the spike operand: integer counts, else SDT-V3's k / 4
                sa, sb = spike_counts(a), spike_counts(b)
                if sa is None and sb is None:
                    sa, sb = spike_counts(a, SDT_LEVELS), spike_counts(b, SDT_LEVELS)
                op, s = (a, sa) if sa is not None else (b, sb)
                self.counter.account(f"matmul{self.calls}", op, out.numel() * a.shape[-1], s)
                self.calls += 1
        return out


def decode_macs(n_queries):
    """SimCC decoding of one query: softmax and expectation over the u and v bins, then the ray cast."""
    from so101wam.percept_net import BINS_U, BINS_V
    return n_queries * (2 * (BINS_U + BINS_V) + 60)


def wrist_frames(ep, wanted):
    cap, out, n = cv2.VideoCapture(str(ep / "wrist.mp4")), {}, 0
    while len(out) < len(wanted):
        ok, bgr = cap.read()
        assert ok, ep
        if n in wanted:
            out[n] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n += 1
    return out


def energy():
    from so101wam.percept_front import SpikeFrontEnd
    from so101wam.scene import Robot
    from spikewam.events import MODES, EventCounter
    from spikewam.text import TextEncoder
    dev = "cuda"
    config = json.loads(CONFIG.read_text())
    rt = WAMRuntime(POLICY, config["instructions"], readout=READOUT, tier="eager", device=dev)
    model = rt.model
    model.cfg.flow_steps = STEPS
    fe = SpikeFrontEnd(PERCEPT, device=dev, cup_ckpt=CUP)
    cube, cup = Count(fe.model), Count(fe.cup_net, sdt_prefix="backbone.")          # one counter per network (hooks fire on every call)
    start_cubes = {m: defaultdict(float) for m in ("dense", "delta")}
    counter = EventCounter(model)
    text = TextEncoder(dev, pool=model.cfg.text_pool, encoder=model.cfg.text_encoder)
    robot = Robot()
    locator = fe.locator(robot)
    fe.estimate = count_calls(fe.estimate, cube)
    fe.estimate_cups = count_calls(fe.estimate_cups, cup)
    gen = torch.Generator(device=dev).manual_seed(0)
    n_plans, n_redetect, episodes, text_ops = 0, 0, [], []
    for ep in sorted(p for p in (RUN / "episodes").iterdir() if p.is_dir()):
        rows = [json.loads(line) for line in open(ep / "frames.jsonl")]
        reqs = [json.loads(line) for line in open(ep / "inference.jsonl")]
        lang = json.loads((ep / "metadata.json").read_text())["task"]
        idx = {r["step"]: rows[r["step"]]["cameras"]["wrist"]["video_frame_index"] for r in reqs}
        frames = wrist_frames(ep, set(idx.values()))
        sop, mac = text.model.ops(**text.tok([lang], return_tensors="pt").to(dev))
        text_ops.append((float(sop), float(mac)))
        tok, tmask = rt.text[lang]
        cube.prev, cup.prev = {}, {}
        logged = reqs[0]["scene"]["start"]
        n_ep_plans, n_ep_redetect = 0, 0
        for r in reqs:
            q = np.asarray(r["anchor_state"], np.float64)
            frame = frames[idx[r["step"]]]
            if r.get("new_episode"):
                before = {m: dict(v) for m, v in cube.tot.items()}
                tracker, info = fe.scene(frame, lang, robot, q)     # cubes of the instruction and both cups
                for m in start_cubes:
                    for key in ("sop", "mac"):
                        start_cubes[m][key] += cube.tot[m][key] - before[m].get(key, 0.0)
                        cube.tot[m][key] = before[m].get(key, 0.0)
                desc = tracker.descriptors()
                gap = np.linalg.norm(np.subtract(info["cup_xy"], logged["cup_xy"])) * 100
            before = cube.calls
            tracker.observe(frame, q, locator)                      # re-detection while the cube is on the table
            n_ep_redetect += cube.calls - before
            nodes, T5 = tracker.update(q, r["step"])
            skel = tracker.skeleton(q, nodes, T5)
            n = len(skel)
            rt.encode(skel, q, desc)
            if n_ep_plans == 0:
                counter.new_episode(n)
            cur, ids, mask = rt.s_cur[:, :n], [d[:, :n] for d in rt.s_desc], rt.s_mask[:, :n]   # real nodes only
            for _ in range(K):
                counter.new_sample()
                noise = torch.randn(1, model.cfg.horizon, model.cfg.action_dim, device=dev, generator=gen)
                with torch.no_grad():
                    model.sample_actions(cur, *ids, mask, tok, tmask, k=1, h=10, units=rt.units, noise=noise)
            n_ep_plans += 1
        n_plans += n_ep_plans
        n_redetect += n_ep_redetect
        episodes.append({"episode": ep.name, "plans": n_ep_plans, "cube_redetections": n_ep_redetect,
                         "events": [list(e) for e in tracker.events], "cup_xy_vs_logged_cm": round(float(gap), 3)})
        print(ep.name, n_ep_plans, "plans", n_ep_redetect, "re-detections", tracker.events, f"cup vs log {gap:.2f} cm",
              flush=True)
    cube.add_mac(decode_macs(n_redetect))
    for m in start_cubes:
        start_cubes[m]["mac"] += decode_macs(3 * len(episodes))
    cup.add_mac(decode_macs(2 * len(episodes)))
    pol = counter.result(n_plans, 8)
    n_ep = len(episodes)
    sop_t, mac_t = np.mean(text_ops, 0)
    res = {"policy": str(POLICY), "readout": READOUT, "percept": str(PERCEPT), "cup": str(CUP), "trials": n_ep,
           "plans": n_plans, "cube_redetections": n_redetect, "episodes": episodes,
           "params": {"policy": sum(p.numel() for p in model.parameters()),
                      "cube_front_end": sum(p.numel() for p in fe.model.parameters()),
                      "cup_sdt_v3": sum(p.numel() for p in fe.cup_net.parameters()),
                      "text": sum(p.numel() for p in text.model.parameters())}}
    for mode in MODES:
        res[f"policy_{mode}"] = {"sop": pol[mode]["sop"], "mac": pol[mode]["mac"],
                                 "uJ_per_plan": pol[mode]["energy_uJ_per_replan"]}
    for mode in ("dense", "delta"):
        res[f"cube_per_plan_{mode}"] = cube.uj(mode, n_plans)
        res[f"cube_per_call_{mode}"] = cube.uj(mode, max(n_redetect, 1))
        t = start_cubes[mode]
        res[f"start_cubes_per_episode_{mode}"] = {"sop": t["sop"] / n_ep, "mac": t["mac"] / n_ep,
                                                  "uJ": (t["sop"] * E_SOP + t["mac"] * E_MAC) * 1e6 / n_ep}
        res[f"start_cup_sdt_per_episode_{mode}"] = cup.uj(mode, n_ep)
    res["text_per_episode"] = {"sop": sop_t, "mac": mac_t, "uJ": (sop_t * E_SOP + mac_t * E_MAC) * 1e6}
    d = res["policy_delta"]["uJ_per_plan"] + res["cube_per_plan_delta"]["uJ"]
    once = res["start_cubes_per_episode_delta"]["uJ"] + res["start_cup_sdt_per_episode_delta"]["uJ"]
    res["summary_uJ"] = {"policy_per_plan": res["policy_delta"]["uJ_per_plan"],
                         "cube_front_end_per_plan": res["cube_per_plan_delta"]["uJ"],
                         "policy_plus_cube_per_plan": d,
                         "start_front_ends_per_episode": once,
                         "text_per_episode": res["text_per_episode"]["uJ"],
                         "all_amortized_per_plan": d + (once + res["text_per_episode"]["uJ"]) * n_ep / n_plans}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "energy.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary_uJ"], indent=1))


class count_calls:
    """Wrap a front-end method: count its network's operations (and attention products) on every call."""

    def __init__(self, fn, counter):
        self.inner, self.counter = fn, counter

    def __call__(self, *args, **kwargs):
        self.counter.calls += 1
        with MatmulCount(self.counter):
            return self.inner(*args, **kwargs)


# ---- speed and memory ----------------------------------------------------------------------------------------------
def val_requests():
    import av
    import pyarrow.parquet as pq
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(VAL / "meta/tasks.jsonl")}
    for path in sorted(glob.glob(str(VAL / "data/chunk-000/*.parquet"))):
        e = int(Path(path).stem.split("_")[-1])
        t = pq.read_table(path, columns=["observation.state", "task_index"])
        state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
        lang = tasks[int(t["task_index"][0].as_py())]
        out = []
        with av.open(str(VAL / f"videos/chunk-000/observation.images.wrist/episode_{e:06d}.mp4")) as c:
            for i, f in enumerate(c.decode(video=0)):
                if i % PERIOD == 0 and i < len(state):
                    out.append((i, state[i], np.ascontiguousarray(f.to_ndarray(format="rgb24"))))
        yield e, lang, out


def process_memory_mib():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                   text=True).strip().splitlines()
    return next((int(r.split(",")[1]) for r in rows if r and int(r.split(",")[0]) == os.getpid()), None)


def speed():
    from serve import Server
    config = json.loads(CONFIG.read_text())
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    server = Server(config, str(POLICY), READOUT, "fused", OUT / "serve_log", "spiking", str(PERCEPT), str(CUP))
    startup = time.perf_counter() - t0
    health = server.health()
    calls, firsts = [], []
    for e, lang, reqs in val_requests():
        for n, (i, state, frame) in enumerate(reqs):
            header = {"protocol": config["protocol"], "lang": lang, "state": state.tolist(), "step": i,
                      "new_episode": n == 0, "image_encoding": "raw"}
            begin = time.perf_counter()
            r = server.infer(header, frame.tobytes())
            wall = 1000 * (time.perf_counter() - begin)
            row = {"episode": e, "tick": i, "inference_ms": r["inference_ms"], "perception_ms": r["perception_ms"],
                   "frontend_ms": r["frontend_ms"], "call_ms": wall}
            (firsts if n == 0 else calls).append(row)
        print(f"episode {e}: {len(reqs)} requests", flush=True)
    stat = lambda v: {"median": float(np.median(v)), "mean": float(np.mean(v)), "sd": float(np.std(v)),
                      "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95))}
    inf = np.array([c["inference_ms"] for c in calls])
    per = np.array([c["perception_ms"] for c in calls])
    res = {"gpu": torch.cuda.get_device_name(), "health": health, "startup_s": startup, "n_calls": len(calls),
           "n_episodes": len(firsts), "server_inference_ms": stat(inf), "perception_ms": stat(per),
           "policy_ms (inference - perception)": stat(inf - per), "call_ms": stat([c["call_ms"] for c in calls]),
           "redetection_share": float(np.mean(per > 0.5)),
           "first_request_frontend_ms": stat([f["frontend_ms"] for f in firsts]),
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "process_memory_mib": process_memory_mib(),
           "calls": calls, "firsts": firsts}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speed.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("calls", "firsts", "health")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["energy", "speed"], required=True)
    a = p.parse_args()
    energy() if a.part == "energy" else speed()
