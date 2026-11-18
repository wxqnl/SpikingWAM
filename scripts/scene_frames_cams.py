"""Camera matrices and object kinds of the scene_frames.py frames, as the perception front end reads them in closed loop
(FrontEnd.views: base-to-pixel matrix of each camera at 256 x 256; SkeletonExtractor.kinds).

The frames of scene_frames.py are restored from the same demonstration states (same task, same first kept demo, same
steps), so the matrices belong to the rendered images; the interventions there move objects, lights or materials, never
the robot or the cameras, so one matrix per frame serves its interventions too. The skeleton of the restored state is
compared with the saved one as a check.

  source setup/env.sh; $SW_PY scripts/scene_frames_cams.py   ->  $SW_DATA/scene_frames/<suite>_<task id>_cams.npz
"""

import argparse
import glob
import os

import h5py
import numpy as np

from spikewam.libero_tasks import bddl_path, get_suite, make_env, scene_objects
from spikewam.perception import IMAGE
from spikewam.skeleton import SkeletonExtractor

CAMS = ("agentview", "robot0_eye_in_hand")


def main():
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default=os.environ.get("SW_DEMOS"))
    p.add_argument("--frames", default=os.path.join(os.environ.get("SW_DATA", "data"), "scene_frames"))
    a = p.parse_args()
    for path in sorted(glob.glob(os.path.join(a.frames, "*.npz"))):
        if path.endswith("_cams.npz"):
            continue
        z = np.load(path)
        suite_name, task_id = str(z["suite"]), int(z["task_id"])
        task = get_suite(suite_name).get_task(task_id)
        env = make_env(bddl_path(task))
        env.reset()
        sim = env.env.sim
        names, ooi = scene_objects(env)
        ext = SkeletonExtractor(sim, names, ooi)
        with h5py.File(os.path.join(a.demos, f"{suite_name}_no_noops_t_256", f"{task.name}_demo.hdf5"), "r") as f:
            d = f["data"][min(f["data"], key=lambda k: int(k.split("_")[-1]))]
            states = d["states"][()]
        mats, gap = [], []
        for t in z["t"]:
            env.set_init_state(states[t])
            data = sim.data._data
            R, pos = data.xmat[ext.base].reshape(3, 3), data.xpos[ext.base]
            world_from_base = np.r_[np.c_[R, pos], [[0, 0, 0, 1]]]
            mats.append([(get_camera_transform_matrix(sim, c, IMAGE, IMAGE) @ world_from_base)[:3] for c in CAMS])
            gap.append(float(np.abs(ext(sim) - z["skel"][len(mats) - 1]).max()))
        env.close()
        assert list(ext.node_id) == list(z["node_id"]), path
        out = path[:-4] + "_cams.npz"
        np.savez(out, M=np.array(mats, np.float32), kinds=np.asarray(ext.kinds), skel_gap=np.array(gap))
        print(f"{os.path.basename(path)}: {len(mats)} frames, skeleton gap max {max(gap) * 100:.4f} cm", flush=True)


if __name__ == "__main__":
    main()
