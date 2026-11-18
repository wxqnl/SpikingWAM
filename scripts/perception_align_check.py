"""Do the stored demo images, the camera matrices and the skeleton truth agree, and does a fresh render match the store?

For a few tasks: replay one demo, render both cameras from the simulator state, compare with the stored frames under the
four flips, project the object nodes with the world-to-pixel matrix and save an overlay. No training.

  source setup/env.sh; MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" $SW_PY scripts/perception_align_check.py
"""

import json
import os

import h5py
import numpy as np
from PIL import Image, ImageDraw

from spikewam.libero_tasks import bddl_path, get_suite, scene_objects
from spikewam.skeleton import N_ROBOT, SkeletonExtractor

SIZE = 256
CAMS = (("agentview", "agentview_rgb"), ("robot0_eye_in_hand", "eye_in_hand_rgb"))
FLIPS = {"none": lambda x: x, "ud": lambda x: x[::-1], "lr": lambda x: x[:, ::-1], "ud+lr": lambda x: x[::-1, ::-1]}
OUT = "runs/perception/checks"


def main():
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    res = []
    for suite_name, task_id in (("libero_spatial", 6), ("libero_goal", 3), ("libero_10", 2)):
        task = get_suite(suite_name).get_task(task_id)
        env = OffScreenRenderEnv(bddl_file_name=bddl_path(task), camera_heights=SIZE, camera_widths=SIZE)
        env.reset()
        sim = env.env.sim
        names, ooi = scene_objects(env)
        ext = SkeletonExtractor(sim, names, ooi)
        path = os.path.join(os.environ["SW_DEMOS"], f"{suite_name}_no_noops_t_256", f"{task.name}_demo.hdf5")
        with h5py.File(path, "r") as f:
            d = f["data"][min(f["data"], key=lambda k: int(k.split("_")[-1]))]
            states = d["states"][()]
            stored = {k: d["obs"][k][()] for _, k in CAMS}
        row = {"task": f"{suite_name}_{task_id:02d}", "n": len(states), "cams": {}}
        for cam, key in CAMS:
            t = len(states) // 2
            obs = env.regenerate_obs_from_state(states[t])
            raw = obs[f"{cam}_image"]
            diffs = {n: float(np.abs(fn(raw).astype(float) - stored[key][t].astype(float)).mean()) for n, fn in FLIPS.items()}
            best = min(diffs, key=diffs.get)
            M = get_camera_transform_matrix(sim, cam, SIZE, SIZE)
            pts = ext.world_points(sim.data._data)
            h = M @ np.c_[pts, np.ones(len(pts))].T
            uv = (h[:2] / h[2]).T
            row["cams"][cam] = {"diff_by_flip": diffs, "best": best}
            upright = raw[::-1].copy()
            im = Image.fromarray(upright).resize((512, 512))
            dr = ImageDraw.Draw(im)
            for i, (u, v) in enumerate(uv):
                c = (0, 255, 255) if i < N_ROBOT else (255, 60, 60)
                dr.ellipse([2 * u - 3, 2 * v - 3, 2 * u + 3, 2 * v + 3], outline=c, width=2)
            im.save(f"{OUT}/overlay_{row['task']}_{cam}.png")
            if cam == "agentview":
                M0 = M.copy()
        for s in (0, 1, len(states) // 3, len(states) - 1):
            env.regenerate_obs_from_state(states[s])
            row.setdefault("agent_M_change", []).append(float(np.abs(get_camera_transform_matrix(sim, "agentview", SIZE, SIZE) - M0).max()))
        res.append(row)
        env.close()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
