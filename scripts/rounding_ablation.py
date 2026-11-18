"""Which spike rounding costs the action precision? (implementation record 4.8)

For a trained spiking checkpoint, the rounding of one group of neurons is switched off at inference (the current is
still scaled by the threshold and clipped at +-D, only round() is skipped), and the action errors of
scripts/loss_breakdown.py are measured on the same fixed samples. Groups: the expert-layer neurons by name (in1, q,
k, v, o, in2, cq, co, in3, hid), "adapt" (hidden spikes of the coordinate / noisy-action / noisy-future adapters),
"head" (spikes read by the output heads), and "all". No retraining, so a group whose rounding matters shows a
lower error without it.

  python scripts/rounding_ablation.py --ckpt runs/s1_both_D4/ckpt_60000.pt
"""

import argparse
import json
import os
import types

from loss_breakdown import load, measure, setup, show
from spikewam.spiking import IntSpike

GROUPS = ("none", "adapt", "head", "in1", "q", "k", "v", "o", "in2", "cq", "co", "in3", "hid", "all")


def smooth_forward(self, current):
    x = current / self.log_threshold.exp()
    return x.clamp(-self.levels if self.bipolar else 0, self.levels)


def neurons(model, group):
    if group == "none":
        return []
    if group == "all":
        return [m for m in model.modules() if isinstance(m, IntSpike)]
    if group == "adapt":
        return [m.spike for m in (model.cur_in, model.fut_in, model.act_in)]
    if group == "head":
        return list(model.head_spike)
    return [e.n[group] for layer in model.layers for e in layer.experts if isinstance(e.n[group], IntSpike)]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--groups", default=",".join(GROUPS))
    pa.add_argument("--batches", type=int, default=20)
    pa.add_argument("--batch", type=int, default=64)
    a = pa.parse_args()
    ds, batches, noise = setup(a.batches, a.batch)
    model, text_all, text_mask_all = load(a.ckpt, ds)
    results = {}
    for group in a.groups.split(","):
        mods = neurons(model, group)
        for m in mods:
            m.forward = types.MethodType(smooth_forward, m)
        results[group] = measure(model, text_all, text_mask_all, batches, noise, [n.split("/")[0] for n in ds.task_names])
        show(f"no rounding: {group} ({len(mods)} neurons)", results[group])
        for m in mods:
            del m.forward                 # back to IntSpike.forward
    out = os.path.join(os.path.dirname(a.ckpt), "rounding_ablation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
