"""Replay the MuJoCo states of the LIBERO demos and save the sim-state skeleton of every step.

Input: the OpenVLA-regenerated demos ($SW_DEMOS/<suite>_no_noops_t_256/<task name>_demo.hdf5; no-op steps removed,
failed replays dropped), which store the flattened simulator state of every step.
Output: $SW_DATA/skeletons/<suite>/<task id>.npz per task with
  skel (steps, nodes, 3) float32, robot base frame, metres; actions (steps, 7) float32;
  ep_start, ep_len (episodes,); node_id, role, slot (nodes,); edges (E, 2); language; obj_of_interest.
The grip site is checked against the demo's recorded end-effector position (mean and max error; the first step of
each demo is left out: its state was copied from the original file while its observation was not).

  python scripts/extract_skeletons.py [--suites libero_spatial,...] [--workers 10]
"""

import argparse
import os
from multiprocessing import Pool

import h5py
import numpy as np

from spikewam.libero_tasks import SUITES, bddl_path, get_suite, make_env, scene_objects
from spikewam.skeleton import SkeletonExtractor


def extract_task(args):
    suite_name, task_id, demo_dir, out_dir = args
    task = get_suite(suite_name).get_task(task_id)
    path = os.path.join(demo_dir, f"{task.name}_demo.hdf5")
    env = make_env(bddl_path(task))
    env.reset()
    names, ooi = scene_objects(env)
    ext = SkeletonExtractor(env.sim, names, ooi)
    grip = ext.robot_sites[0]
    skel, actions, starts, lens, errs = [], [], [], [], []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in demos:
            d = f["data"][k]
            states, acts, ee = d["states"][()], d["actions"][()], d["obs/ee_pos"][()]
            starts.append(sum(lens))
            lens.append(len(acts))
            for t in range(len(acts)):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                skel.append(ext(env.sim))
                if t > 0:
                    errs.append(np.abs(env.sim.data._data.site_xpos[grip] - ee[t]).max())
            actions.append(acts.astype(np.float32))
    env.close()
    os.makedirs(os.path.join(out_dir, suite_name), exist_ok=True)
    out = os.path.join(out_dir, suite_name, f"{task_id:02d}.npz")
    np.savez_compressed(out, skel=np.stack(skel), actions=np.concatenate(actions),
                        ep_start=np.asarray(starts), ep_len=np.asarray(lens),
                        node_id=ext.node_id, role=ext.role, slot=ext.slot, edges=ext.edges,
                        language=task.language, obj_of_interest=np.asarray(ooi))
    return suite_name, task_id, len(demos), sum(lens), ext.n_nodes, float(np.mean(errs)), float(np.max(errs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--demos", default=os.environ.get("SW_DEMOS"))
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_DATA", "data"), "skeletons"))
    a = p.parse_args()
    jobs = [(s, t, os.path.join(a.demos, f"{s}_no_noops_t_256"), a.out)
            for s in a.suites.split(",") for t in range(get_suite(s).n_tasks)]
    with Pool(a.workers) as pool:
        for s, t, n_ep, n_steps, n_nodes, mean_err, max_err in pool.imap_unordered(extract_task, jobs):
            print(f"{s}/{t:02d}: {n_ep} demos, {n_steps} steps, {n_nodes} nodes, "
                  f"grip-site error mean {mean_err:.1e} max {max_err:.1e} m", flush=True)


if __name__ == "__main__":
    main()
