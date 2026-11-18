"""Forward kinematics of the aloha-agilex arms in torch: joint vector (B, 14) -> the 20 robot nodes of the skeleton
(per arm: base_link and link1..link6 origins, grip site, the two finger tips; spikewam.robotwin.layout), for joint
perturbations whose skeleton stays consistent (implementation record 4.13).

The chain (joint origins and axes, fl_joint1..8 / fr_joint1..8) is read from the embodiment URDF. The world pose of
each arm base and the finger travel as a function of the gripper value are fitted to recorded skeletons (fit()), so the
nodes come out in the frame and units of the data; fit() reports the residual.

  python -m spikewam.robotwin.fk --data /mnt/data/user01/spikewam_robotwin/demo_clean    # fit, check, save
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

URDF = "/mnt/data/user01/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
ARMS = ("fl", "fr")
N_ARM_NODES = 10                                       # base_link, link1..link6, grip site, two finger tips


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def read_chain(arm, urdf=URDF):
    """[(origin 4x4, axis (3,), type)] of joint1..joint8 of one arm."""
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().iter("joint")}
    out = []
    for k in range(1, 9):
        j = joints[f"{arm}_joint{k}"]
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rot(rpy), xyz
        out.append((T, np.array([float(v) for v in j.find("axis").get("xyz").split()]), j.get("type")))
    return out


def _axis_rot(axis, q):
    """(B, 3, 3) rotation by angles q (B,) about a unit axis (Rodrigues)."""
    a = torch.as_tensor(axis / np.linalg.norm(axis), dtype=q.dtype, device=q.device)
    K = torch.zeros(3, 3, dtype=q.dtype, device=q.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
    s, c = torch.sin(q)[:, None, None], torch.cos(q)[:, None, None]
    return torch.eye(3, dtype=q.dtype, device=q.device) + s * K + (1 - c) * (K @ K)


class ArmFK:
    def __init__(self, params=None, device="cpu"):
        self.chains = {arm: read_chain(arm) for arm in ARMS}
        self.params = params                  # per arm: base 4x4 (world <- arm base), finger travel (a, b) per gripper
        self.device = device

    def base_frame(self, q6, grip_travel, arm):
        """(B, 10, 3) nodes in the arm-base frame for joints q6 (B, 6) and finger travel (B,) (metres)."""
        B, dt = q6.shape[0], q6.dtype
        T = torch.eye(4, dtype=dt, device=q6.device).expand(B, 4, 4).clone()
        pts = [T[:, :3, 3]]
        chain = self.chains[arm]
        for k in range(6):
            o, ax, _ = chain[k]
            T = T @ torch.as_tensor(o, dtype=dt, device=q6.device)
            R = torch.eye(4, dtype=dt, device=q6.device).expand(B, 4, 4).clone()
            R[:, :3, :3] = _axis_rot(ax, q6[:, k])
            T = T @ R
            pts.append(T[:, :3, 3])
        tips = []
        for k in (6, 7):
            o, ax, _ = chain[k]
            off = torch.as_tensor(o[:3, 3], dtype=dt, device=q6.device) + grip_travel[:, None] * torch.as_tensor(
                ax, dtype=dt, device=q6.device)
            tips.append((T[:, :3, :3] @ off[..., None])[..., 0] + T[:, :3, 3])
        pts += [(tips[0] + tips[1]) / 2] + tips
        return torch.stack(pts, 1)

    def __call__(self, q):
        """(B, 20, 3) world-frame robot nodes for joint vectors q (B, 14) (radians, gripper value)."""
        out = []
        for a, arm in enumerate(ARMS):
            p = self.params[arm]
            travel = p["travel"][0] + p["travel"][1] * q[:, 7 * a + 6]
            nodes = self.base_frame(q[:, 7 * a:7 * a + 6], travel, arm)
            base = torch.as_tensor(p["base"], dtype=q.dtype, device=q.device)
            out.append(nodes @ base[:3, :3].T + base[:3, 3])
        return torch.cat(out, 1)


def kabsch(src, dst):
    """Rigid transform (4x4) mapping src (N, 3) onto dst (N, 3) in least squares."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, md - R @ ms
    return T


def load_pairs(data, n=20000, seed=0):
    """(q (n, 14), robot nodes (n, 20, 3)) from recorded frames: q_t = the action of the previous frame."""
    qs, nodes = [], []
    for path in sorted(glob.glob(os.path.join(data, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        for st, ln in zip(d["ep_start"], d["ep_len"]):
            qs.append(d["actions"][st:st + ln - 1])
            nodes.append(d["skel"][st + 1:st + ln, :2 * N_ARM_NODES])
    q, x = np.concatenate(qs), np.concatenate(nodes)
    pick = np.random.default_rng(seed).choice(len(q), min(n, len(q)), replace=False)
    return q[pick].astype(np.float64), x[pick].astype(np.float64)


def fit(data, n=20000):
    """Base pose of each arm (Kabsch on the 8 nodes that do not depend on the fingers) and the finger travel as a
    linear function of the gripper value (least squares on the tip positions); returns params and residuals (mm)."""
    q, x = load_pairs(data, n)
    fk, params, res = ArmFK(), {}, {}
    qt = torch.as_tensor(q)
    for a, arm in enumerate(ARMS):
        sl = slice(a * N_ARM_NODES, (a + 1) * N_ARM_NODES)
        nb = fk.base_frame(qt[:, 7 * a:7 * a + 6], torch.zeros(len(q), dtype=qt.dtype), arm).numpy()
        base = kabsch(nb[:, :8].reshape(-1, 3), x[:, sl][:, :8].reshape(-1, 3))
        # finger travel: project the world tip offset from the grip site onto the finger axis in the world
        n0 = fk.base_frame(qt[:, 7 * a:7 * a + 6], torch.ones(len(q), dtype=qt.dtype), arm).numpy()
        tip_w = (x[:, sl][:, 8] - x[:, sl][:, 7]) - (nb[:, 8] - nb[:, 7]) @ base[:3, :3].T   # minus the zero-travel offset
        axis_w = ((n0[:, 8] - n0[:, 7]) - (nb[:, 8] - nb[:, 7])) @ base[:3, :3].T     # tip offset per metre of travel
        travel = (tip_w * axis_w).sum(-1) / (axis_w * axis_w).sum(-1)
        A = np.stack([np.ones(len(q)), q[:, 7 * a + 6]], 1)
        coef = np.linalg.lstsq(A, travel, rcond=None)[0]
        params[arm] = {"base": base.tolist(), "travel": coef.tolist()}
    fk.params = params
    err = np.linalg.norm(fk(qt).numpy() - x, axis=-1) * 1000
    res = {"mean_mm": float(err.mean()), "p99_mm": float(np.percentile(err, 99)), "max_mm": float(err.max()),
           "per_node_mean_mm": err.mean(0).round(3).tolist()}
    return params, res


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data", default="/mnt/data/user01/spikewam_robotwin/demo_clean")
    pa.add_argument("--n", type=int, default=20000)
    a = pa.parse_args()
    params, res = fit(a.data, a.n)
    print(json.dumps(res, indent=1))
    with open(os.path.join(a.data, "fk_params.json"), "w") as f:
        json.dump(params, f, indent=1)
    print("wrote", os.path.join(a.data, "fk_params.json"))


if __name__ == "__main__":
    main()
