"""Summary of scripts/rt_factor_check.py: front-end error at the first plan under Easy, Hard and Hard minus one factor.

Per scene: mean node error (cm, as the closed loop's percept_cm, after the rigid refit); per object slot the centre
error, its part along the head camera's ray (depth) and across it, the height error, the rotation error (Kabsch,
slots of 3+ nodes) and whether the centre is off by more than 5 cm (another object taken for it). Hard minus table
height and Hard minus clutter keep Hard's random stream, so they are also compared scene by scene with Hard.

  python scripts/rt_factor_summary.py runs/perception/checks/factor_rtB30tCM [--weak 10]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

VARIANTS = ["easy", "hard", "hard_no_table", "hard_no_clutter", "hard_no_background", "hard_no_light"]


def rot_deg(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    if np.linalg.matrix_rank(P, tol=1e-4) < 2:
        return np.nan
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def scenes(root, variant):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, variant, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        task = os.path.basename(f)[:-4]
        for i, seed in enumerate(d["seed"]):
            tr, es = np.asarray(d["true"][i], np.float64), np.asarray(d["est"][i], np.float64)
            slot = np.asarray(d["slot"][i])
            M = np.asarray(d["head"][i], np.float64)
            C = -np.linalg.solve(M[:, :3], M[:, 3])
            node = np.linalg.norm(es - tr, axis=1).mean() * 100
            objs = []
            for s in np.unique(slot):
                j = slot == s
                ct, ce = tr[j].mean(0), es[j].mean(0)
                e = (ce - ct) * 100
                u = (ct - C) / np.linalg.norm(ct - C)
                depth = float(e @ u)
                objs.append(dict(centre=float(np.linalg.norm(e)), depth=depth, across=float(np.linalg.norm(e - depth * u)),
                                 dz=float(e[2]), rot=rot_deg(tr[j], es[j]) if j.sum() >= 3 else np.nan))
            out[(task, int(seed))] = dict(node=node, objs=objs, tz=float(d["table_z_bias"][i]) * 100)
    return out


def row(name, sc):
    if not sc:
        return f"| {name} | 0 | | | | | | | |"
    node = np.array([v["node"] for v in sc.values()])
    o = [x for v in sc.values() for x in v["objs"]]
    c = np.array([x["centre"] for x in o])
    ok = c <= 5
    dep = np.abs([x["depth"] for x in o])[ok]
    acr = np.array([x["across"] for x in o])[ok]
    dz = np.abs([x["dz"] for x in o])[ok]
    rot = np.array([x["rot"] for x in o])[ok]
    return (f"| {name} | {len(sc)} | {node.mean():.2f} / {np.median(node):.2f} | {np.median(c):.2f} / {np.percentile(c, 90):.2f} | "
            f"{100 * (~ok).mean():.1f}% | {np.median(dep):.2f} | {np.median(acr):.2f} | {np.median(dz):.2f} | {np.nanmedian(rot):.1f}° |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--weak", type=int, default=10, help="also the first this many tasks of factor_tasks.txt (largest Hard gap)")
    a = p.parse_args()
    order = open(os.path.join(os.path.dirname(a.root.rstrip("/")), "factor_tasks.txt")).read().split()
    data = {v: scenes(a.root, v) for v in VARIANTS}
    head = ("| 设置 | 场景数 | 节点误差均值 / 中位（cm） | 物体中心误差中位 / 90%（cm） | 中心偏 >5 cm（认错） | "
            "视线方向（深度）误差中位 | 垂直视线误差中位 | 高度误差中位 | 转角误差中位 |\n|---|---|---|---|---|---|---|---|---|")
    for title, keep in (("全部任务", None), (f"Hard 差距最大的 {a.weak} 个任务", set(order[:a.weak]))):
        print(f"\n{title}（深度、垂直、高度、转角只算中心偏 5 cm 以内的物体）\n\n{head}")
        for v in VARIANTS:
            print(row(v, {k: x for k, x in data[v].items() if keep is None or k[0] in keep}))
    # scene by scene: Hard vs Hard minus table height / clutter (same objects, textures and light)
    print("\n同一场景对比（Hard 与去掉一项后，物体、纹理、光照相同）\n")
    print("| 去掉 | 配对场景 | Hard 节点误差均值 | 去掉后 | 变化 | 变好的场景比例 |\n|---|---|---|---|---|---|")
    for v in ("hard_no_table", "hard_no_clutter"):
        keys = sorted(set(data["hard"]) & set(data[v]))
        if not keys:
            continue
        h = np.array([data["hard"][k]["node"] for k in keys])
        x = np.array([data[v][k]["node"] for k in keys])
        print(f"| {v[len('hard_no_'):]} | {len(keys)} | {h.mean():.2f} | {x.mean():.2f} | {x.mean() - h.mean():+.2f} | {100 * (x < h).mean():.0f}% |")
    # height error against the table offset (Hard): does the front end see the table height?
    o = [(v["tz"], x["dz"]) for v in data["hard"].values() for x in v["objs"] if x["centre"] <= 5]
    if len(o) > 10:
        tz, dz = np.array(o).T
        k, b = np.polyfit(tz, dz, 1)
        r = np.corrcoef(tz, dz)[0, 1]
        print(f"\nHard 的高度误差对桌面偏移（cm）：dz = {k:.2f} × 偏移 {b:+.2f}，r = {r:.2f}，{len(tz)} 个物体"
              f"（斜率 −1 = 完全没看到桌高，0 = 看到了）")
    o = [(x["dz"]) for v in data["hard_no_table"].values() for x in v["objs"] if x["centre"] <= 5]
    if o:
        print(f"Hard 去掉桌高（偏移 0）：高度误差均值 {np.mean(o):+.2f} cm，{len(o)} 个物体")
    # per task (hard vs easy vs hard_no_table)
    print("\n逐任务（节点误差均值 cm，按 Hard 差距排序）\n\n| 任务 | Easy | Hard | Hard 去桌高 | Hard 去杂物 | Hard 去背景 | Hard 去光照 |\n|---|---|---|---|---|---|---|")
    for t in order:
        cells = []
        for v in VARIANTS:
            vals = [x["node"] for k, x in data[v].items() if k[0] == t]
            cells.append(f"{np.mean(vals):.2f}" if vals else "")
        if any(cells):
            print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
