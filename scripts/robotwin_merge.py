"""Merge the parts of a split RoboTwin collection (spikewam/robotwin/collect.py --seed0 --tag) into one <task>.npz.

  python scripts/robotwin_merge.py --parts /mnt/data/user01/spikewam_robotwin/demo_clean/parts \
      --out /mnt/data/user01/spikewam_robotwin/demo_clean --task put_object_cabinet [--episodes 50]
Parts are read in seed order (<task>.p<seed0>.npz) and the first --episodes episodes are kept.
"""

import argparse
import glob
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=50)
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.parts, f"{a.task}.p*.npz")), key=lambda f: int(f.split(".p")[-1][:-4]))
    eps = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        for e, (st, ln, n) in enumerate(zip(z["ep_start"], z["ep_len"], z["n_nodes"])):
            eps.append({"skel": z["skel"][st:st + ln, :n], "actions": z["actions"][st:st + ln], "n": n,
                        "desc": [z[k][e, :n] for k in ("node_id", "role", "slot")], "seed": z["seeds"][e],
                        "language": z["language"][e], "meta": z["meta"][e]})
    eps = eps[:a.episodes]
    assert len(eps) == a.episodes, f"{a.task}: only {len(eps)} episodes in {files}"
    n_pad = max(e["n"] for e in eps)
    lens = np.array([len(e["actions"]) for e in eps])
    st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    skel = np.zeros((lens.sum(), n_pad, 3), np.float32)
    desc = np.zeros((3, len(eps), n_pad), np.int64)
    for i, e in enumerate(eps):
        skel[st[i]:st[i] + lens[i], :e["n"]] = e["skel"]
        for k in range(3):
            desc[k, i, :e["n"]] = e["desc"][k]
    np.savez_compressed(
        os.path.join(a.out, f"{a.task}.npz"), skel=skel, actions=np.concatenate([e["actions"] for e in eps]),
        n_nodes=np.array([e["n"] for e in eps]), node_id=desc[0], role=desc[1], slot=desc[2], ep_start=st,
        ep_len=lens, seeds=np.array([e["seed"] for e in eps]),
        language=np.array([e["language"] for e in eps], dtype=object),
        meta=np.array([e["meta"] for e in eps], dtype=object))
    print(f"{a.task}: {len(eps)} episodes from {len(files)} parts, {lens.sum()} frames")


if __name__ == "__main__":
    main()
