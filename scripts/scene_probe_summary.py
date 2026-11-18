"""Tables of the skeleton-vs-camera probes (scene_probe_image.py, scene_probe_skeleton.py, follow_gain.py).

  python scripts/scene_probe_summary.py [--image base=runs/scene_probe_image_base turbovla=runs/scene_probe_image_turbovla]
                                        [--skeleton runs/scene_probe_skeleton]

Action response: mean over the first 12 actions of the L2 change of the translation command (env units, [-1, 1] per
axis) between an intervention and its frame, averaged over interventions of a kind; "same" is the frame's second copy
in the batch. Ratio: response to a kind / response to a task-object move.
"""

import argparse
import json

import numpy as np

KINDS = ("same", "task", "other", "light", "tint")
REGIONS = ("background", "robot", "task_objects", "other_objects")


def responses(stem):
    r, z = json.load(open(stem + ".json")), np.load(stem + ".npz")
    meta = r["meta"]
    kinds = np.array([iv["kind"] for m in meta for iv in m["iv"]])
    fr = np.array([i for i, m in enumerate(meta) for _ in m["iv"]])
    closed = np.array([m["closed"] for m in meta])
    move = lambda a, b: np.linalg.norm(a[..., :12, :3] - b[..., :12, :3], axis=-1).mean(-1)
    d = move(z["iv_actions"], z["actions"][fr])
    out = {"same": float(move(z["same_actions"], z["actions"]).mean()), "n_frames": len(meta)}
    for k in KINDS[1:]:
        out[k] = float(d[kinds == k].mean())
        out[k + "_n"] = int((kinds == k).sum())
        for ph, sel in (("free", ~closed[fr]), ("holding", closed[fr])):
            out[f"{k}_{ph}"] = float(d[(kinds == k) & sel].mean()) if ((kinds == k) & sel).any() else None
    out["action_scale"] = float(np.linalg.norm(z["actions"][:, :12, :3], axis=-1).mean())
    return r, z, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", nargs="*", default=["spike_vla_base=runs/scene_probe_image_base",
                                                  "turbovla=runs/scene_probe_image_turbovla"])
    p.add_argument("--skeleton", default="runs/scene_probe_skeleton")
    p.add_argument("--follow", default="runs/f2tm16_geo_finecur16_textmean/follow_gain_ckpt_60000_ema_k1_steps1_fixed.json")
    p.add_argument("--out", default="runs/scene_probe_summary.json")
    a = p.parse_args()
    summary = {"response": {}, "events": {}, "attention": {}}
    print("== action response (translation command change, env units; ratio to a task-object move)")
    print(f"{'model':16s} " + " ".join(f"{k:>8s}" for k in KINDS) + "   other/task light/task tint/task")
    models = [s.split("=") for s in a.image] + [["spikewam", a.skeleton]]
    for name, stem in models:
        try:
            r, z, out = responses(stem)
        except FileNotFoundError:
            continue
        summary["response"][name] = out
        t = out["task"]
        print(f"{name:16s} " + " ".join(f"{out[k]:8.4f}" for k in KINDS) +
              f"   {out['other'] / t:9.2f} {out['light'] / t:10.2f} {out['tint'] / t:9.2f}")
        if "attn" in z.files and z["attn"].shape[1] == 4:               # image: (frames, regions) visual mass
            att = z["attn"] / z["attn"].sum(1, keepdims=True)
            summary["attention"][name] = {"share": dict(zip(REGIONS, att.mean(0).tolist())),
                                          "area": dict(zip(REGIONS, z["area"].mean(0).tolist()))}
        elif "attn" in z.files:                                         # skeleton: robot, task nodes, actions
            w = z["attn"]
            nodes = w[:, :2].sum(1, keepdims=True)
            summary["attention"][name] = {"share_of_node_attention": dict(zip(("robot", "task_objects"),
                                                                              (w[:, :2] / nodes).mean(0).tolist())),
                                          "on_nodes": float(nodes.mean()), "on_action_tokens": float(w[:, 2].mean())}
        if "events" in r:
            ev = {}
            for mode, e in r["events"].items():
                uj = np.array(e["energy_uJ_by_cam_region"])                # (cams, regions)
                ev[mode] = {"total_uJ": float(uj.sum()), "share": dict(zip(REGIONS, (uj.sum(0) / uj.sum()).tolist())),
                            "share_agentview": dict(zip(REGIONS, (uj[0] / uj[0].sum()).tolist())),
                            "share_wrist": dict(zip(REGIONS, (uj[1] / uj[1].sum()).tolist()))}
            summary["events"][name] = ev
        if "events_uJ_per_frame" in r:
            summary["events"][name] = {m: {"total_uJ": sum(v.values()), "share": {k: x / sum(v.values())
                                                                                   for k, x in v.items()}}
                                       for m, v in r["events_uJ_per_frame"].items()}
            summary["contact"] = r.get("contact")
    print("\n== events")
    print(json.dumps(summary["events"], indent=1))
    print("\n== attention")
    print(json.dumps(summary["attention"], indent=1))
    if summary.get("contact"):
        print("\n== world-model contact test")
        print(json.dumps(summary["contact"], indent=1))
    try:
        summary["follow_gain"] = json.load(open(a.follow))
        g = summary["follow_gain"]["gain transport (landed within 5 cm)"]
        print(f"\n== following gain (transport, landed within 5 cm): SpikeWAM {g['gain']:.2f} "
              f"[{g['ci'][0]:.2f}, {g['ci'][1]:.2f}] n={g['n']}; Spike-VLA base 0.24 [0.21, 0.26], TurboVLA 0.68 "
              f"(spike_vla_icml/research_plan.md, same protocol)")
    except FileNotFoundError:
        pass
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
