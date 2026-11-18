"""Residuals of a RoboTwin front end on the held-out demonstrations, per task: the input noise of spikewam.train --input_noise.

For every task the residuals (estimate - truth, metres) of at most MAX_FRAMES held-out frames, indexed by node id, so that the
errors of one frame stay together (a common shift of an object, a confused identity).

  CUDA_VISIBLE_DEVICES=6 python scripts/rt_residuals.py --ckpt runs/perception/rt160b/ckpt.pt
Writes <ckpt dir>/residuals.npz: tasks (T), count (T), res (T, F, N_NODE_IDS, 3).
"""

import argparse
import importlib.util
import os
import re

import numpy as np
import torch

from spikewam.perception import PerceptConfig, SpikePercept
from spikewam.robotwin.layout import N_NODE_IDS

MAX_FRAMES = 800


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception_rt"))
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("train_perception", os.path.join(here, "train_perception.py"))
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    dev = torch.device("cuda")
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PerceptConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st["cfg"].items()})
    model = SpikePercept(cfg).to(dev).eval()
    model.load_state_dict(st["model"])
    data = tp.DataRT(a.root, dev, "spikeminilm16")
    assert tuple(data.kinds) == tuple(cfg.kinds), "another list of object kinds than the checkpoint's"
    names = [re.sub(r"^demo_(clean|randomized)_", "", n) for n in data.names]
    tasks = sorted(set(names))
    index = {n: i for i, n in enumerate(tasks)}
    of_task = np.array([index[n] for n in names])[data.task_np[data.val_idx]]
    res, count = np.zeros((len(tasks), MAX_FRAMES, N_NODE_IDS, 3), np.float32), np.zeros(len(tasks), np.int64)
    for t in range(len(tasks)):
        idx = data.val_idx[of_task == t]
        idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), MAX_FRAMES)).astype(int)]
        count[t] = len(idx)
        for k, images in data.loader(idx, 256, 4):
            i = torch.as_tensor(idx[k.numpy()], device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(tp.prepare(images, cfg.size, dev, False), data.M[i], data.text[data.task[i], 0], data.ids[i], data.roles[i], data.grip[i], data.kind[i])
            e, mask = (out["fused"].float() - data.skel[i]) * data.mask[i][..., None], data.ids[i].cpu().numpy()
            for j, frame in enumerate(k.numpy()):
                res[t, frame, mask[j]] = e[j].cpu().numpy()
        print(tasks[t], count[t], flush=True)
    out = os.path.join(os.path.dirname(a.ckpt), "residuals.npz")
    np.savez(out, tasks=np.array(tasks), count=count, res=res)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
