"""Speed and peak GPU memory of the real-robot baselines (X-VLA, pi0.5) on the SO-101 validation demos (11-13).

The baselines' server code (zhenji/so101_deploy/baselines/serve.py, inputs.py) was deleted by the 11-13 source cleanup.
It is rebuilt in real_robot/efficiency/restore/so101_deploy/baselines from the bytecode cache the server left
(__pycache__/serve.cpython-310.pyc, 11-09 02:53, the version behind the 11-09 speed table): compiled under Python 3.10,
all 15 code objects match the cache instruction for instruction. The 34 deleted openpi files are copied from
/home/user01/Minko/openpi after a SHA-256 check against SOURCE-CLEANUP.tsv. Weights, X-VLA code and assets are symlinks
to the original bundle; the cleaned folder is not touched.

Protocol (same as check.py on 11-09 and scripts/real_robot_efficiency.py --part speed): the 18 validation demos, a
request every 9th tick, the collector client's own request (two views, external crop, 224 x 224), the server's own
inference_ms; the HTTP hop is replaced by a direct call to Runtime.infer. Peak memory over the whole process
(load, two warm-up calls, all requests). The normalized actions are compared with the 11-09 server log.

    cd real_robot/efficiency/restore/so101_deploy/baselines   # as serve.sh does
    <env python> scripts/real_robot_baseline_efficiency.py xvla|pi05
"""
import glob
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parent / "zhenji/so101_deploy"
RESTORE = ROOT / "real_robot/efficiency/restore/so101_deploy/baselines"
OUT = ROOT / "real_robot/efficiency"
DATA = DEPLOY / "checks/val_episodes"
OLD = DEPLOY / "checks/speed_20261109"
PERIOD = 9


def frames(episode, views):
    import av
    streams = [av.open(str(DATA / f"videos/chunk-000/observation.images.{v}/episode_{episode:06d}.mp4")) for v in views]
    try:
        for decoded in zip(*(c.decode(video=0) for c in streams)):
            yield {v: f.to_ndarray(format="rgb24") for v, f in zip(views, decoded)}
    finally:
        for c in streams:
            c.close()


def process_memory_mib():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                   text=True).strip().splitlines()
    return next((int(r.split(",")[1]) for r in rows if r and int(r.split(",")[0]) == os.getpid()), None)


def main(model):
    import pyarrow.parquet as pq
    import torch
    os.environ["SO101_MODEL"] = model
    sys.path.insert(0, str(DEPLOY / "collector"))
    from hippovla import HippoVLA, load_config
    sys.path.insert(0, str(RESTORE))
    import serve
    config = json.loads((RESTORE / "serve_config.json").read_text())
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    runtime = serve.Runtime(SimpleNamespace(model=model, checkpoint=None, denoise_steps=10), config)
    startup = time.perf_counter() - t0
    after_load = {"reserved_gib": torch.cuda.memory_reserved() / 2**30, "allocated_gib": torch.cuda.memory_allocated() / 2**30}

    class Local(HippoVLA):                          # the collector client, the HTTP hop replaced by a direct call
        def _call(self, method, path, body=None):
            if path == "/health":
                return json.loads(json.dumps(runtime.health()))
            size = struct.unpack("<I", body[:4])[0]
            return json.loads(json.dumps(runtime.infer(json.loads(body[4:4 + size]), body[4 + size:]), allow_nan=False))

    client_config = load_config()
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(DATA / "meta/tasks.jsonl")}
    files = sorted(glob.glob(str(DATA / "data/chunk-000/*.parquet")))
    views = sorted(set(client_config["views"]) | {"wrist"})
    calls, firsts, outputs = [], [], []
    for e in range(len(files)):
        t = pq.read_table(files[e], columns=["observation.state", "task_index"])
        state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
        task = tasks[t["task_index"][0].as_py()]
        client = Local(config=client_config)
        replies, call = [], client._call

        def recorded(method, path, body=None):
            reply = call(method, path, body)
            replies.append(reply)
            return reply
        client._call = recorded
        n = 0
        for i, fr in enumerate(frames(e, views)):
            if i >= len(state):
                break
            client.observe(i, fr)
            if i % PERIOD:
                continue
            client.predict(client.example(state[i], task, i))
            r = replies[-1]
            outputs.append(r["normalized_actions"])
            (firsts if n == 0 else calls).append({"episode": e, "tick": i, "inference_ms": r["inference_ms"]})
            n += 1
        client.close()
        print(f"episode {e}: {n} requests", flush=True)

    old = [json.loads(l) for l in open(OLD / f"check_{model}.jsonl")]
    same_order = len(old) == len(outputs)
    diff = [float(np.abs(np.asarray(a) - np.asarray(b["normalized_actions"])).max()) for a, b in zip(outputs, old)]
    old_ms = np.array([o["inference_ms"] for o in old if not o["new_episode"]])
    stat = lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "median": float(np.median(v)),
                      "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95))}
    inf = np.array([c["inference_ms"] for c in calls])
    res = {"model": model, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "python": sys.version.split()[0],
           "precision": "weights FP32, forward under torch.autocast(cuda, bfloat16), TF32 matmul on (restored serve.py)",
           "checkpoint": str(runtime.checkpoint), "global_step": runtime.step, "denoise_steps": runtime.steps,
           "startup_s": startup, "n_calls": len(calls), "n_episodes": len(firsts),
           "server_inference_ms": stat(inf), "first_request_ms": stat([f["inference_ms"] for f in firsts]),
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
           "after_load": after_load, "process_memory_mib": process_memory_mib(),
           "params": sum(p.numel() for p in runtime.model.parameters()),
           "param_bytes": sum(p.numel() * p.element_size() for p in runtime.model.parameters()),
           "check_vs_20261109": {"requests_match": same_order, "n": len(diff),
                                 "max_abs_diff_normalized_actions": {"median": float(np.median(diff)), "p95": float(np.percentile(diff, 95)),
                                                                     "max": float(np.max(diff))},
                                 "old_server_inference_ms": stat(old_ms)},
           "calls": calls, "firsts": firsts}
    (OUT / f"baseline_{model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("calls", "firsts")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
