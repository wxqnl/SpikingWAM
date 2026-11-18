"""Rendered LIBERO scenes and physical interventions, the shared input of the skeleton-vs-camera probes
(scene_probe_image.py, scene_probe_skeleton.py).

For the first kept demonstration of every LIBERO task, every --stride-th step (the replanning points of energy.py), the
demo's MuJoCo state is restored and both cameras are rendered at 256 x 256 with instance segmentation, rotated as the
image policies see them (180 degrees). Saved per frame: the two images, a region map per camera (0 background:
table, floor, walls and everything outside the scene objects; 1 robot; 2 task objects, the BDDL obj_of_interest;
3 other objects), the robot state of the LIBERO image policies (eef position, quaternion, gripper qpos), the
skeleton, and whether the demo closes the gripper at that step.

Interventions on the same state (the images, robot state and skeleton are rendered again):
  task:<name>   a movable task object moved 2 cm in xy (fixed random direction per frame and object);
  other:<name>  a movable non-task object moved 2 cm in the same way;
  light         every light's diffuse and specular intensity halved;
  tint          the background materials (table, floor, walls) tinted blue-grey.
Objects within 6 cm of the grip site are not moved (held or being grasped). The scene is not stepped after an
intervention (open-loop probes).

  python scripts/scene_frames.py [--stride 16] [--workers 20]   ->  $SW_DATA/scene_frames/<suite>_<task id>.npz

Several EGL renderers on one GPU occasionally abort inside mjr_readPixels; run one process per GPU (--workers 1,
GPU chosen by CUDA_VISIBLE_DEVICES) and rerun a task that aborted.
"""

import argparse
import os
import multiprocessing as mp

import h5py
import numpy as np

from spikewam.libero_tasks import SUITES, bddl_path, get_suite, scene_objects
from spikewam.skeleton import SkeletonExtractor

ROBOT_INSTANCES = ("MountedPanda0", "Panda0", "RethinkMount0", "PandaGripper0")
SHIFT, NEAR_GRIP = 0.02, 0.06
TINT = np.array([0.55, 0.65, 0.85, 1.0])
CAMS = ("agentview", "robot0_eye_in_hand")


def rot(x):
    return np.ascontiguousarray(x[::-1, ::-1])


def build(task):
    from libero.libero.envs.env_wrapper import SegmentationRenderEnv
    return SegmentationRenderEnv(bddl_file_name=bddl_path(task), camera_heights=256, camera_widths=256,
                                 camera_segmentations="instance")


def do_task(args):
    suite_name, task_id, demo_dir, out_dir, stride = args
    task = get_suite(suite_name).get_task(task_id)
    env = build(task)
    env.reset()
    inner, sim = env.env, env.env.sim
    model = sim.model._model
    names, ooi = scene_objects(env)
    ext = SkeletonExtractor(sim, names, ooi)
    # region of every segmentation instance (render value = instance index + 1, 0 = no instance)
    inst = list(inner.model.instances_to_ids.keys())
    task_objs = {n for n in names if any(o == n or o.startswith(n + "_") for o in ooi)}
    region = np.zeros(len(inst) + 1, np.uint8)
    for i, n in enumerate(inst):
        region[i + 1] = 1 if n in ROBOT_INSTANCES else (2 if n in task_objs else 3)
    movable = [n for n in inner.objects_dict if inner.objects_dict[n].joints]
    qadr = {n: sim.model.get_joint_qpos_addr(inner.objects_dict[n].joints[0])[0] for n in movable}
    bg_geoms = sorted(set(range(model.ngeom)) - set(inner.model.geom_ids_to_instances))
    in_inst = {int(model.geom_matid[g]) for g in inner.model.geom_ids_to_instances}
    bg_mats = sorted({int(model.geom_matid[g]) for g in bg_geoms} - in_inst - {-1})
    bg_plain = [g for g in bg_geoms if model.geom_matid[g] == -1]
    light0 = (model.light_diffuse.copy(), model.light_specular.copy())
    mat0, geom0 = model.mat_rgba.copy(), model.geom_rgba.copy()
    grip = ext.robot_sites[0]

    def render(state):
        obs = env.regenerate_obs_from_state(state)
        out = {"rgb": np.stack([rot(obs[f"{c}_image"]) for c in CAMS]),
               "robot_state": np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                              obs["robot0_gripper_qpos"]]).astype(np.float32),
               "skel": ext(sim).astype(np.float32)}
        return out, obs

    path = os.path.join(demo_dir, f"{suite_name}_no_noops_t_256", f"{task.name}_demo.hdf5")
    with h5py.File(path, "r") as f:
        d = f["data"][min(f["data"], key=lambda k: int(k.split("_")[-1]))]     # failed replays were dropped
        states, actions = d["states"][()], d["actions"][()]
    rng = np.random.default_rng(1000 * SUITES.index(suite_name) + task_id)
    frames, iv = [], []
    for t in range(0, len(states), stride):
        base, obs = render(states[t])
        base["region"] = np.stack([region[rot(obs[f"{c}_segmentation_instance"][..., 0])] for c in CAMS])
        base["t"], base["closed"] = t, bool(actions[t, -1] > 0)
        frames.append(base)
        fi = len(frames) - 1
        gpos = sim.data._data.site_xpos[grip].copy()
        for n in movable:
            if np.linalg.norm(sim.data.body_xpos[inner.obj_body_id[n]] - gpos) < NEAR_GRIP:
                continue
            ang = rng.uniform(0, 2 * np.pi)
            s = states[t].copy()
            s[1 + qadr[n]:1 + qadr[n] + 2] += SHIFT * np.array([np.cos(ang), np.sin(ang)])
            r, _ = render(s)
            iv.append({"frame": fi, "kind": "task" if n in task_objs else "other", "object": n, **r})
        model.light_diffuse[:] = light0[0] * 0.5
        model.light_specular[:] = light0[1] * 0.5
        r, _ = render(states[t])
        iv.append({"frame": fi, "kind": "light", "object": "", **r})
        model.light_diffuse[:], model.light_specular[:] = light0
        model.mat_rgba[bg_mats] = mat0[bg_mats] * TINT
        model.geom_rgba[bg_plain] = geom0[bg_plain] * TINT
        r, _ = render(states[t])
        iv.append({"frame": fi, "kind": "tint", "object": "", **r})
        model.mat_rgba[:], model.geom_rgba[:] = mat0, geom0
    env.close()
    st = lambda rows, k: np.stack([r[k] for r in rows])
    out = os.path.join(out_dir, f"{suite_name}_{task_id:02d}.npz")
    np.savez_compressed(
        out, suite=suite_name, task_id=task_id, language=task.language, obj_of_interest=np.asarray(ooi),
        node_id=ext.node_id, role=ext.role, slot=ext.slot,
        rgb=st(frames, "rgb"), region=st(frames, "region"), robot_state=st(frames, "robot_state"),
        skel=st(frames, "skel"), t=st(frames, "t"), closed=st(frames, "closed"),
        iv_frame=np.array([r["frame"] for r in iv]), iv_kind=np.array([r["kind"] for r in iv]),
        iv_object=np.array([r["object"] for r in iv]), iv_rgb=st(iv, "rgb"), iv_robot_state=st(iv, "robot_state"),
        iv_skel=st(iv, "skel"))
    return suite_name, task_id, len(frames), len(iv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default=os.environ.get("SW_DEMOS"))
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_DATA", "data"), "scene_frames"))
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--tasks", default="")
    p.add_argument("--workers", type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for s in a.suites.split(","):
        ids = [int(x) for x in a.tasks.split(",")] if a.tasks else range(get_suite(s).n_tasks)
        jobs += [(s, t, a.demos, a.out, a.stride) for t in ids]
    if a.workers == 1:      # in-process: a render that aborts ends the run (a pool would wait for the lost task)
        for r in map(do_task, jobs):
            print(*r, flush=True)
        return
    with mp.get_context("spawn").Pool(a.workers) as pool:     # EGL contexts do not survive fork
        for r in pool.imap_unordered(do_task, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
