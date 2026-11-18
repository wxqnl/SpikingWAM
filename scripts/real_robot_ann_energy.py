"""FLOPs and energy per plan of the two real-robot baselines in their SO-101 setting (paper 5.4, 11-13).

Same rule and layer arithmetic as paper/figures/vla_energy_estimate.py (one forward pass for one action chunk, image
encoding and all denoising steps included, 4.6 pJ per MAC, normalisation not counted). SO-101 setting from the
fine-tuning recipes (zhenji/so101_deploy/baselines/runs/<model>/recipe.json) and zhenji/baselines/README.md: external
and wrist images at 224 x 224, a 16-step chunk, 10 denoising steps.
  X-VLA   num_actions 16, 32 soft prompts, Florence-2 config as in the paper's LIBERO estimate;
  pi0.5   action_horizon 16, max_token_len 200 (prompt padded, state as discrete tokens), Gemma 2B + 300M expert.
Only the two real views are counted (the serving adapters, deleted in the 11-13 source cleanup, could not be checked
for a padded third view, which openpi and the X-VLA LIBERO client otherwise encode); the 3-slot count is printed too.
Output: real_robot/efficiency/ann_energy.json
"""
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/figures"))
with contextlib.redirect_stdout(io.StringIO()):          # the module prints its LIBERO report on import
    import vla_energy_estimate as V


def xvla_so101(slots):
    n_image, n_text, n_action, n_prompt, d = 50, 50, 16, 32, 1024
    n_vlm = n_image + n_text
    n_aux = (slots - 1) * n_image
    n_all = n_action + n_vlm + n_aux + n_prompt
    per_step = V.vit(n_all, d, 4 * d, 24) + (n_vlm + n_aux) * d * d + n_action * (20 + 32 + 20) * d + n_action * d * 20
    return {"Florence DaViT, 2 views at 224": 2 * (V.davit_224() + n_image * 2048 * d),
            f"BART encoder, {n_vlm} tokens": V.vit(n_vlm, d, 4 * d, 12),
            f"transformer, {n_all} tokens, 10 steps": 10 * per_step}


def pi05_so101(slots):
    n_prefix = slots * 256 + 200
    return {f"SigLIP So400m/14, {slots} views": V.siglip_so400m(slots),
            f"Gemma 2B, {n_prefix} tokens": V.vit(n_prefix, 2048, 16384, 18),
            "action expert, 10 steps": 10 * 18 * V.expert_layer(16, n_prefix, 1024, 4096)}


res = {}
for name, fn in (("X-VLA", xvla_so101), ("pi0.5", pi05_so101)):
    for slots in (2, 3):
        parts = fn(slots)
        macs = sum(parts.values())
        res[f"{name}, {slots} image slots"] = {"GMACs": macs / 1e9, "GFLOPs": 2 * macs / 1e9, "mJ": macs * 4.6e-12 * 1e3,
                                               "parts_GMACs": {k: v / 1e9 for k, v in parts.items()}}
        print(f"{name}, {slots} slots: {macs / 1e9:,.1f} GMACs, {2 * macs / 1e9:,.0f} GFLOPs, {macs * 4.6e-9:,.1f} mJ")
(ROOT / "real_robot/efficiency/ann_energy.json").write_text(json.dumps(res, indent=1))
