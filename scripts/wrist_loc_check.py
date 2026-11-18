"""How well the wrist camera alone can place LIBERO's movable task objects, by the geometry of the SO-101 front end.

On the first demonstration of every task, every --stride-th step: the object's instance mask in the wrist view (the
simulator's segmentation, in place of the real robot's colour blob) gives the mask centroid; its pixel is cast onto the
horizontal plane through the object's top face with the camera pose of that step, and the result is compared with the
object's true centre. No policy is run.

  source setup/env.sh; MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" $SW_PY scripts/wrist_loc_check.py
"""

import argparse
import json
import multiprocessing as mp
import os

import h5py
import numpy as np

from spikewam.libero_tasks import SUITES, bddl_path, get_suite, scene_objects
from spikewam.skeleton import ROLES, SkeletonExtractor

SIZE = 256
MIN_PIXELS = 30              # the real robot's 150 px at 640 x 480, at this resolution
CAMERA = "robot0_eye_in_hand"


def cast(M, u, v, z):
    """Pixel (u, v) of the upright image onto the plane at height z, with the world-to-pixel matrix M."""
    a = np.array([M[0, :2] - u * M[2, :2], M[1, :2] - v * M[2, :2]])
    b = -np.array([(M[0, 2] - u * M[2, 2]) * z + M[0, 3] - u * M[2, 3], (M[1, 2] - v * M[2, 2]) * z + M[1, 3] - v * M[2, 3]])
    return np.linalg.solve(a, b)


def do_task(args):
    from libero.libero.envs.env_wrapper import SegmentationRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    suite_name, task_id, demo_dir, stride = args
    task = get_suite(suite_name).get_task(task_id)
    env = SegmentationRenderEnv(bddl_file_name=bddl_path(task), camera_heights=SIZE, camera_widths=SIZE, camera_segmentations="instance")
    env.reset()
    inner, sim = env.env, env.env.sim
    names, ooi = scene_objects(env)
    ext = SkeletonExtractor(sim, names, ooi)
    inst = list(inner.model.instances_to_ids.keys())
    objects = []
    for m, entry in enumerate(ooi):
        obj, _ = ext._resolve(entry, names)
        if obj in inner.objects_dict and inner.objects_dict[obj].joints:
            nodes = np.where(ext.slot == m + 1)[0]
            role = ext.role[nodes]
            objects.append((obj, inst.index(obj) + 1, nodes[role == ROLES["center"]][0], nodes[role == ROLES["top"]][0]))
    path = os.path.join(demo_dir, f"{suite_name}_no_noops_t_256", f"{task.name}_demo.hdf5")
    with h5py.File(path, "r") as f:
        d = f["data"][min(f["data"], key=lambda k: int(k.split("_")[-1]))]
        states, actions = d["states"][()], d["actions"][()]
    rows, z0 = [], {}
    grip = ext.robot_sites[0]
    for t in range(0, len(states), stride):
        obs = env.regenerate_obs_from_state(states[t])
        seg = obs[f"{CAMERA}_segmentation_instance"][..., 0][::-1]     # upright, as the world-to-pixel matrix expects
        M = get_camera_transform_matrix(sim, CAMERA, SIZE, SIZE)
        pts = ext.world_points(sim.data._data)
        gpos = sim.data._data.site_xpos[grip].copy()
        for name, value, center, top in objects:
            z0.setdefault(name, float(pts[center, 2]))
            mask = seg == value
            n = int(mask.sum())
            row = {"task": f"{suite_name}_{task_id:02d}", "t": t, "object": name, "pixels": n, "closed": bool(actions[t, -1] > 0),
                   "lifted_cm": float(pts[center, 2] - z0[name]) * 100, "grip_cm": float(np.linalg.norm(pts[center, :2] - gpos[:2])) * 100}
            p = M @ np.r_[pts[top], 1]
            row["projected_inside"] = bool(0 <= p[0] / p[2] < SIZE and 0 <= p[1] / p[2] < SIZE and p[2] > 0)
            if n >= MIN_PIXELS:
                v, u = np.where(mask)
                row["touches_border"] = bool(u.min() == 0 or v.min() == 0 or u.max() == SIZE - 1 or v.max() == SIZE - 1)
                xy = cast(M, float(u.mean()), float(v.mean()), float(pts[top, 2]))
                row["error_cm"] = float(np.linalg.norm(xy - pts[center, :2])) * 100
                row["reprojection_px"] = float(np.hypot(p[0] / p[2] - u.mean(), p[1] / p[2] - v.mean()))
            rows.append(row)
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default=os.environ.get("SW_DEMOS"))
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_RUNS", "runs"), "wrist_loc_check.json"))
    a = p.parse_args()
    jobs = [(s, t, a.demos, a.stride) for s in SUITES for t in range(get_suite(s).n_tasks)]
    jobs = jobs[:a.limit] if a.limit else jobs
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = [r for out in pool.imap_unordered(do_task, jobs) for r in out]
    json.dump(rows, open(a.out, "w"))
    table = [r for r in rows if r["lifted_cm"] < 1.0 and not (r["closed"] and r["grip_cm"] < 6)]
    seen = [r for r in table if "error_cm" in r]
    whole = [r for r in seen if not r["touches_border"]]
    first = [r for r in table if r["t"] == 0]
    q = lambda x: {"n": len(x), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)), "within_1cm": float(np.mean(np.array(x) < 1)),
                   "within_2.5cm": float(np.mean(np.array(x) < 2.5))} if len(x) else None
    res = {"rows": len(rows), "objects_on_table": len(table), "seen": len(seen) / max(len(table), 1), "seen_whole": len(whole) / max(len(table), 1),
           "first_frame": {"objects": len(first), "seen": float(np.mean(["error_cm" in r for r in first])), "projected_inside": float(np.mean([r["projected_inside"] for r in first]))},
           "reprojection_px_whole": q([r["reprojection_px"] for r in whole]), "error_cm_seen": q([r["error_cm"] for r in seen]), "error_cm_whole": q([r["error_cm"] for r in whole]),
           "by_grip_distance": {}}
    for lo, hi in ((0, 6), (6, 12), (12, 20), (20, 200)):
        sub = [r for r in table if lo <= r["grip_cm"] < hi]
        sw = [r for r in sub if "error_cm" in r]
        res["by_grip_distance"][f"{lo}-{hi}cm"] = {"n": len(sub), "seen": len(sw) / max(len(sub), 1), "error": q([r["error_cm"] for r in sw])}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
