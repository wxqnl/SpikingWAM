"""Skeletons of the language-annotated CALVIN segments (spikewam/calvin/skeleton.py).

Every segment of <split>/lang_annotations/auto_lang_ann.npy becomes one entry with its own instruction. The entry holds
the segment's frames and, after them, up to --ext more frames of the same recording (the play continues after the
annotated window): training starts samples only inside the segment (n_sample frames) but reads the action chunk and
the future skeletons from the following frames, so that chunks near the segment end are the recorded motion rather
than the last action repeated.

  python scripts/calvin_extract.py --split /mnt/.../task_ABC_D/training --out data/calvin/task_ABC_D/training
  python scripts/calvin_extract.py --split /mnt/.../task_ABC_D/validation --scene calvin_scene_D --out ...

Writes <out>/<scene>.npz: skel (F, 53, 3), actions (F, 7) = rel_actions, robot_obs (F, 15), scene_obs (F, 24), and per
entry ep_start, ep_len, n_sample, frame0 (episode id of its first frame), language, task; node_id, role, slot (53,).
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SK = {}


def skeleton_of(split, scene):
    if scene not in _SK:
        from spikewam.calvin.skeleton import CalvinSkeleton, make_env
        _SK[scene] = CalvinSkeleton(make_env(split, scene))
    return _SK[scene]


def job(args):
    split, scene, segs = args
    sk = skeleton_of(split, scene)
    out = []
    for s, last in segs:
        ro, so, act = [], [], []
        for f in range(s, last + 1):
            with np.load(os.path.join(split, f"episode_{f:07d}.npz")) as z:
                ro.append(z["robot_obs"])
                so.append(z["scene_obs"])
                act.append(z["rel_actions"])
        ro, so = np.array(ro), np.array(so)
        out.append((sk(ro, so), np.array(act, np.float32), ro.astype(np.float32), so.astype(np.float32)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scene", default="", help="the split's only scene (validation has no scene_info.npy)")
    p.add_argument("--ext", type=int, default=32, help="frames kept after each segment (chunk / future horizon)")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--per_job", type=int, default=40)
    a = p.parse_args()
    ann = np.load(os.path.join(a.split, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()
    eps = np.load(os.path.join(a.split, "ep_start_end_ids.npy"))
    if a.scene:
        scenes = {a.scene: (int(eps.min()), int(eps.max()))}
    else:
        scenes = np.load(os.path.join(a.split, "scene_info.npy"), allow_pickle=True).item()
    segs = {sc: [] for sc in scenes}
    for i, (s, e) in enumerate(ann["info"]["indx"]):
        sc = next(k for k, (lo, hi) in scenes.items() if lo <= s <= hi)
        ep_end = int(eps[(eps[:, 0] <= s) & (s <= eps[:, 1]), 1][0])
        assert e <= ep_end and ep_end <= scenes[sc][1]
        segs[sc].append((i, int(s), int(e), min(int(e) + a.ext, ep_end)))
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        for sc, items in segs.items():
            jobs = [(a.split, sc, [(s, last) for _, s, _, last in items[k:k + a.per_job]])
                    for k in range(0, len(items), a.per_job)]
            res = [r for chunk in pool.map(job, jobs, chunksize=1) for r in chunk]
            lens = np.array([len(r[0]) for r in res])
            sk = skeleton_of(a.split, sc)
            np.savez(os.path.join(a.out, f"{sc}.npz"),
                     skel=np.concatenate([r[0] for r in res]), actions=np.concatenate([r[1] for r in res]),
                     robot_obs=np.concatenate([r[2] for r in res]), scene_obs=np.concatenate([r[3] for r in res]),
                     ep_start=np.concatenate([[0], np.cumsum(lens)[:-1]]), ep_len=lens,
                     n_sample=np.array([e - s + 1 for _, s, e, _ in items]), frame0=np.array([s for _, s, _, _ in items]),
                     language=np.array([ann["language"]["ann"][i] for i, *_ in items]),
                     task=np.array([ann["language"]["task"][i] for i, *_ in items]),
                     node_id=sk.node_id, role=sk.role, slot=sk.slot)
            print(f"{sc}: {len(items)} segments, {lens.sum()} frames ({sum(e - s + 1 for _, s, e, _ in items)} "
                  f"sampled)", flush=True)


if __name__ == "__main__":
    main()
