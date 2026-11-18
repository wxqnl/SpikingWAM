"""Render the two cameras of the LIBERO demos for the spiking keypoint front end (spikewam/perception.py).

Replays every demo's simulator states with the renderer of the closed-loop evaluation. Every --stride-th step (the first
step of a demo is skipped, its state was copied from another file) gives, per task under <out>/<suite>_<id>[_s<shard>]:
  rgb.npy   (N, 2, 256, 256, 3) uint8, agentview and wrist, upright (the images the world-to-pixel matrices expect)
  meta.npz  skel (N, n, 3) in the robot base frame; M (N, 2, 3, 4) base-to-pixel matrices at 256 x 256; ep (N,), t (N,);
            node_id, role, slot, kind (n,); language

--dr: domain randomization for robustness. The arena textures (table, floor, walls) are replaced by procedural patterns, the
lights change, and the third-person camera moves over a wide range (the known camera matrix follows). Nothing comes from the
LIBERO-Plus assets.
  source setup/env.sh; MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" $SW_PY scripts/perception_cache.py
"""

import argparse
import multiprocessing as mp
import os
import re
import time

import h5py
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from spikewam.libero_tasks import SUITES, bddl_path, get_suite, scene_objects
from spikewam.skeleton import N_ROBOT, SkeletonExtractor

SIZE = 256
CAMS = ("agentview", "robot0_eye_in_hand")
ARENA = re.compile("wall|floor|table|desk|paint|decor", re.I)
PIVOT = np.array([0.0, 0.0, 0.8])


def fractal(rng, w, h, octaves=5, persistence=0.55, stretch=1):
    """Fractal noise (h, w) in [0, 1]; stretch > 1 elongates it along x."""
    out, amp = np.zeros((h, w), np.float32), 1.0
    for o in range(octaves):
        g = 2 ** (o + 2)
        a = rng.random((g, max(2, int(g / stretch)))).astype(np.float32)
        out += amp * np.asarray(Image.fromarray(a, mode="F").resize((w, h), Image.BICUBIC))
        amp *= persistence
    return (out - out.min()) / max(out.max() - out.min(), 1e-6)


def natural(rng, w, h):
    """A surface with the colours and structure of real materials: stone, marble, wood, tiles or a dark plain, mostly unsaturated."""
    v = rng.uniform(0.02, 0.95) if rng.random() < 0.6 else rng.uniform(0.02, 0.2)
    tone = rng.uniform(-1, 1, 3) * rng.uniform(0, 0.12)
    c1 = np.clip(v * (1 + tone), 0, 1)
    c2 = np.clip(c1 + rng.uniform(-0.35, 0.35), 0, 1)
    kind, i, j = rng.integers(5), *np.meshgrid(np.arange(w) / w, np.arange(h) / h)
    if kind == 0:                      # stone, concrete
        a = fractal(rng, w, h, int(rng.integers(4, 7)))
    elif kind == 1:                    # marble veins
        fx, fy = rng.uniform(1, 8, 2)
        a = (np.sin(2 * np.pi * (i * fx + j * fy) + fractal(rng, w, h) * rng.uniform(3, 12)) + 1) / 2
        a = a ** rng.uniform(0.3, 3)
    elif kind == 2:                    # wood grain
        grain = fractal(rng, w, h, 5, 0.6, stretch=float(rng.uniform(6, 30)))
        a = (np.sin(2 * np.pi * rng.uniform(4, 30) * (j + 0.15 * grain)) + 1) / 2 * 0.5 + grain * 0.5
    elif kind == 3:                    # tiles with grout and a slight tint per tile
        n, line = int(rng.integers(2, 14)), rng.uniform(0.02, 0.08)
        a = fractal(rng, w, h, 4) * 0.4 + 0.3
        p, q = (i * n) % 1, (j * n) % 1
        tint = rng.random((n, n)).astype(np.float32)[(j * n).astype(int) % n, (i * n).astype(int) % n] * 0.3
        a = np.where((p < line) | (q < line), 1.0 if rng.random() < 0.5 else 0.0, a + tint)
    else:                              # dark plain with fine structure
        a = fractal(rng, w, h, 3) * rng.uniform(0.02, 0.3)
    img = (c1[None, None] * (1 - a[..., None]) + c2[None, None] * a[..., None]) * 255
    return np.clip(img + rng.normal(0, rng.uniform(0, 6), (h, w, 1)), 0, 255).astype(np.uint8)


def pattern(rng, w, h, nat=0.0):
    """A procedural RGB texture (h, w, 3): flat, smooth noise, tiles, stripes or stretched grain, with some speckle."""
    if rng.random() < nat:
        return natural(rng, w, h)
    k, c1, c2 = rng.random(), rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    if k < 0.2:
        img = np.full((h, w, 3), c1, np.uint8)
    elif k < 0.5:
        g = int(rng.integers(2, 48))
        img = np.asarray(Image.fromarray(rng.integers(0, 256, (g, g, 3), np.uint8)).resize((w, h), Image.BILINEAR))
    elif k < 0.7:                  # tiles
        n = int(rng.integers(2, 24))
        i, j = np.meshgrid(np.arange(w) * n // w, np.arange(h) * n // h)
        img = np.where(((i + j) % 2 == 0)[..., None], c1, c2).astype(np.uint8)
    elif k < 0.85:                 # stripes
        n, a = int(rng.integers(3, 40)), rng.random() * np.pi
        i, j = np.meshgrid(np.arange(w) / w, np.arange(h) / h)
        img = np.where((np.sin((i * np.cos(a) + j * np.sin(a)) * n * np.pi) > 0)[..., None], c1, c2).astype(np.uint8)
    else:                          # grain: noise stretched along one axis, between two colours
        a = rng.integers(0, 256, (int(rng.integers(32, 128)), 8), np.uint8).astype(np.float32) / 255
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)).astype(np.float32) / 255
        a = a if rng.random() < 0.5 else a.T[:h, :w] if w == h else a
        img = (c1 * a[..., None] + c2 * (1 - a[..., None])).astype(np.uint8)
    if rng.random() < 0.5:
        small = rng.integers(0, 2, (h // 2 + 1, w // 2 + 1), np.int8)[:h // 2, :w // 2].repeat(2, 0).repeat(2, 1) * 2 - 1
        pad = np.zeros((h, w), np.int16)
        pad[:small.shape[0], :small.shape[1]] = small * int(rng.integers(3, 30))
        img = np.clip(img.astype(np.int16) + pad[..., None], 0, 255).astype(np.uint8)
    return img


class Randomizer:
    """Changes one simulator's arena textures, lights and third-person camera in place."""

    def __init__(self, sim, rng, nat=0.0):
        self.sim, self.rng, m = sim, rng, sim.model._model
        self.nat = nat
        self.m = m
        self.data = m.tex_data if hasattr(m, "tex_data") else m.tex_rgb
        tex = set()
        for g in range(m.ngeom):
            mat = m.geom_matid[g]
            name = m.geom(g).name or m.body(m.geom_bodyid[g]).name
            if mat >= 0 and ARENA.search(name):
                t = m.mat_texid[mat]
                t = int(t.max() if np.ndim(t) else t)
                if t >= 0 and m.tex(t).name.startswith("tex"):      # object and fixture textures carry the object's name
                    tex.add(t)
        self.tex = sorted(tex)
        self.cam = m.camera("agentview").id
        self.pos, self.quat = m.cam_pos[self.cam].copy(), m.cam_quat[self.cam].copy()
        self.light = {k: getattr(m, f"light_{k}").copy() for k in ("diffuse", "specular", "ambient", "pos", "dir")}
        self.head = m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy()

    def textures(self):
        m, ctx = self.m, self.sim._render_context_offscreen
        for t in self.tex:
            if self.rng.random() < 0.15:                  # some surfaces keep the last pattern
                continue
            w, h, n = int(m.tex_width[t]), int(m.tex_height[t]), int(getattr(m, "tex_nchannel", np.full(m.ntex, 3))[t])
            img = pattern(self.rng, w, h, self.nat)
            self.data[m.tex_adr[t]:m.tex_adr[t] + w * h * n] = np.dstack([img, np.full((h, w), 255, np.uint8)])[..., :n].ravel()
            mujoco.mjr_uploadTexture(m, ctx.con, t)

    def lights(self):
        m, r = self.m, self.rng
        gain = np.exp(r.uniform(np.log(0.15), np.log(1.8)))
        tint = r.uniform(0.6, 1.0, 3)
        for k in range(m.nlight):
            m.light_diffuse[k] = np.clip(self.light["diffuse"][k] * gain * tint * r.uniform(0.6, 1.4), 0, 2)
            m.light_specular[k] = np.clip(self.light["specular"][k] * r.uniform(0, 2.5), 0, 1)
            m.light_ambient[k] = np.clip(self.light["ambient"][k] * r.uniform(0, 3), 0, 1)
            m.light_pos[k] = self.light["pos"][k] + r.uniform(-1.5, 1.5, 3)
            d = self.light["dir"][k] + r.uniform(-0.5, 0.5, 3)
            m.light_dir[k] = d / max(np.linalg.norm(d), 1e-6)
            m.light_castshadow[k] = r.random() < 0.4
        m.vis.headlight.ambient[:] = np.clip(self.head[0] * r.uniform(0.2, 3), 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(self.head[1] * r.uniform(0.2, 1.8), 0, 1)

    def camera(self):
        """Orbit, height, distance about the workspace and a small pan and tilt, over the range LIBERO-Plus-like variations need."""
        m, r = self.m, self.rng
        if r.random() < 0.25:
            m.cam_pos[self.cam], m.cam_quat[self.cam] = self.pos, self.quat
            return
        az = r.uniform(-80, 80) if r.random() < 0.5 else r.uniform(-25, 25)
        el = r.uniform(-8, 22)
        scale = r.uniform(0.9, 2.1) if r.random() < 0.5 else r.uniform(0.9, 1.3)
        R = Rot.from_quat(self.quat[[1, 2, 3, 0]])
        turn = Rot.from_euler("z", az, degrees=True) * Rot.from_euler("y", -el, degrees=True)
        p = PIVOT + scale * turn.apply(self.pos - PIVOT)
        R = turn * R
        R = Rot.from_euler("z", r.uniform(-13, 13), degrees=True) * Rot.from_euler("y", r.uniform(-13, 13), degrees=True) * R
        q = R.as_quat()
        m.cam_pos[self.cam], m.cam_quat[self.cam] = p, q[[3, 0, 1, 2]]

    def apply(self):
        self.textures()
        self.lights()
        self.camera()

    def draw_objects(self):
        """Per free object: None (stays), or a table-plane shift and a yaw (or None)."""
        m, r = self.m, self.rng
        out = []
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and r.random() < 0.7:
                out.append((int(m.jnt_qposadr[j]), r.uniform(-0.25, 0.25, 2), r.uniform(-np.pi, np.pi) if r.random() < 0.5 else None))
        return out

    def move_objects(self, state, moves):
        """A copy of the flat MuJoCo state (time, qpos, qvel) with the objects moved; objects at the gripper stay (it may hold them)."""
        s = state.copy()
        self.sim.set_state_from_flattened(s)
        self.sim.forward()
        grip = self.sim.data._data.site_xpos[self.m.site("gripper0_grip_site").id]
        for adr, dxy, yaw in moves:
            q = s[1 + adr:8 + adr]                 # x y z qw qx qy qz (a view of s)
            if np.linalg.norm(q[:2] - grip[:2]) < 0.1 and abs(q[2] - grip[2]) < 0.15:
                continue
            x, y = q[:2] + dxy
            if abs(x) > 0.45 or abs(y) > 0.55:     # stays on the workspace
                continue
            q[0], q[1] = x, y
            if yaw is not None:
                q[3:] = (Rot.from_euler("z", yaw) * Rot.from_quat(q[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
        return s


def do_task(args):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    suite_name, task_id, demo_dir, out, stride, per_demo, dr, every, shard, shards, objects, nat = args
    task = get_suite(suite_name).get_task(task_id)
    d = os.path.join(out, f"{suite_name}_{task_id:02d}" + (f"_s{shard}" if shards > 1 else ""))
    if os.path.exists(os.path.join(d, "meta.npz")):
        return d, -1, 0.0
    t0 = time.time()
    os.makedirs(d, exist_ok=True)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path(task), camera_heights=SIZE, camera_widths=SIZE)
    env.reset()
    sim = env.env.sim
    names, ooi = scene_objects(env)
    ext = SkeletonExtractor(sim, names, ooi)
    path = os.path.join(demo_dir, f"{suite_name}_no_noops_t_256", f"{task.name}_demo.hdf5")
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        states = [f["data"][k]["states"][()] for k in demos]
    frames = [(e, t) for e, s in enumerate(states) if e % shards == shard for t in range(1, len(s), stride)]
    rng = np.random.default_rng(zlib_seed(suite_name, task_id, shard, nat) if nat else zlib_seed(suite_name, task_id, shard))
    rand = Randomizer(sim, rng, nat) if dr else None
    last = -1
    rgb = np.lib.format.open_memmap(os.path.join(d, "rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(len(frames), 2, SIZE, SIZE, 3))
    skel, mats = [], []
    for i, (e, t) in enumerate(frames):
        if per_demo and e != last:        # fixtures are placed at reset and are not in the state vector: a new layout per demo
            env.reset()
            sim = env.env.sim
            ext = SkeletonExtractor(sim, names, ooi)
            rand = Randomizer(sim, rng, nat) if dr else None
            last = e
        if dr and i % every == 0:
            rand.apply()
            moves = rand.draw_objects() if objects else None
        state = rand.move_objects(states[e][t], moves) if dr and objects else states[e][t]
        obs = env.regenerate_obs_from_state(state)
        data = sim.data._data
        R, p = data.xmat[ext.base].reshape(3, 3), data.xpos[ext.base]
        world_from_base = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
        mats.append([(get_camera_transform_matrix(sim, c, SIZE, SIZE) @ world_from_base)[:3] for c in CAMS])
        skel.append(ext(sim))
        for c, cam in enumerate(CAMS):
            rgb[i, c] = obs[f"{cam}_image"][::-1]
    rgb.flush()
    del rgb
    os.replace(os.path.join(d, "rgb.tmp.npy"), os.path.join(d, "rgb.npy"))
    np.savez(os.path.join(d, "meta.npz"), skel=np.stack(skel).astype(np.float32), M=np.array(mats, np.float32),
             ep=np.array([e for e, _ in frames]), t=np.array([t for _, t in frames]), node_id=ext.node_id, role=ext.role, slot=ext.slot,
             kind=np.array(ext.kinds), language=task.language)
    assert len(ext.kinds) == len(ext.node_id) - N_ROBOT
    env.close()
    return d, len(frames), time.time() - t0


def zlib_seed(*parts):
    import zlib
    return zlib.crc32(repr(parts).encode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default=os.environ.get("SW_DEMOS"))
    p.add_argument("--out", default=os.path.join(os.environ.get("SW_DATA", "data"), "perception"))
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--per_demo_reset", action="store_true", help="reset the scene before every demo (new fixture layout)")
    p.add_argument("--dr", action="store_true", help="domain randomization of textures, lights and the third-person camera")
    p.add_argument("--every", type=int, default=3, help="--dr: a new draw every this many rendered frames")
    p.add_argument("--shards", type=int, default=1, help="split each task's demos over this many jobs")
    p.add_argument("--objects", action="store_true", help="--dr: also move the free objects over the table and turn them")
    p.add_argument("--natural", type=float, default=0.0, help="--dr: share of the textures drawn as natural materials")
    a = p.parse_args()
    jobs = [(s, t, a.demos, a.out, a.stride, a.per_demo_reset, a.dr, a.every, k, a.shards, a.objects, a.natural)
            for s in SUITES for t in range(get_suite(s).n_tasks) for k in range(a.shards)]
    jobs = jobs[:a.limit] if a.limit else jobs
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for d, n, sec in pool.imap_unordered(do_task, jobs):
            print(f"{os.path.basename(d)} {n} frames {sec:.0f} s {time.strftime('%X')}", flush=True)


if __name__ == "__main__":
    main()
