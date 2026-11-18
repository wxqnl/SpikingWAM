"""CALVIN skeleton from the state observations (robot_obs (15,), scene_obs (24,)): the node layout of
spikewam/calvin/layout.py.

The extractor reads the state of every recorded frame, the evaluator that of the live environment, and both go through
CalvinSkeleton, which holds its own copy of the scene (a CALVIN environment without cameras that is never stepped): the
robot nodes are the forward kinematics of the measured joints in that copy, the block nodes are the block boxes at
the recorded poses, the fixture nodes the link boxes at joint 0 shifted along the joint axis by the recorded state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .layout import (BLOCKS, BOX_NODES, FIXTURE_STATE, FIXTURES, MAX_SLOTS, N_NODES, N_ROBOT, ROBOT_LINKS,
                     ROLES, SLOT_SIZE)

# box node offsets in units of the half extents (center, top, +x, -x, +y, -y)
BOX_DIRS = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], np.float64)


def make_env(split_dir, scene=None):
    """A CALVIN environment with the configuration of a dataset split (the one the official evaluation builds from
    <split>/.hydra/merged_config.yaml; scene D for the validation splits), without cameras; `scene` (e.g.
    "calvin_scene_A") replaces the split's scene. The scene's whole configuration replaces the split's (a merge would
    keep the split's order of the movable objects, and scene_obs lists the blocks in the order of the scene's file:
    A pink, blue, red; B red, blue, pink; C blue, red, pink; D red, blue, pink)."""
    import calvin_env
    import hydra
    from omegaconf import OmegaConf
    conf = Path(calvin_env.__file__).parents[1] / "conf"
    cfg = OmegaConf.load(Path(split_dir) / ".hydra" / "merged_config.yaml")
    cfg.cameras = {}
    if scene:
        new = OmegaConf.load(conf / "scene" / f"{scene}.yaml")
        new.name = scene
        cfg.scene = new
    cfg.env.use_egl = False
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)


def box_nodes(center, half, rot=None):
    """(..., 6, 3) box nodes; rot (..., 3, 3) turns the offsets (block frame), none for a world-aligned box."""
    off = BOX_DIRS * half[..., None, :]
    if rot is not None:
        off = np.einsum("...ij,...nj->...ni", rot, off)
    return center[..., None, :] + off


class CalvinSkeleton:
    def __init__(self, env):
        """env: a CALVIN environment of the scene (make_env) used only as a kinematic copy: joints are reset in it."""
        self.env, self.p, self.cid = env, env.p, env.cid
        robot = env.robot
        self.robot_uid = robot.robot_uid
        names = {self.p.getJointInfo(self.robot_uid, j, physicsClientId=self.cid)[12].decode(): j
                 for j in range(self.p.getNumJoints(self.robot_uid, physicsClientId=self.cid))}
        self.robot_links = [names[n] for n in ROBOT_LINKS]
        self.arm_joints, self.grip_joints = list(robot.arm_joint_ids), list(robot.gripper_joint_ids)
        table = next(o for o in env.scene.fixed_objects if o.name == "table")
        uid = table.uid
        links = {self.p.getJointInfo(uid, j, physicsClientId=self.cid)[12].decode(): j
                 for j in range(self.p.getNumJoints(uid, physicsClientId=self.cid))}
        # fixture boxes at joint 0 and the link displacement per unit of the joint (all four joints are prismatic)
        self.fix_center, self.fix_half, self.fix_axis = [], [], []
        for name in FIXTURES:
            j = links[name]
            assert self.p.getJointInfo(uid, j, physicsClientId=self.cid)[2] == self.p.JOINT_PRISMATIC, name
            q0 = self.p.getJointState(uid, j, physicsClientId=self.cid)[0]
            self.p.resetJointState(uid, j, 0.0, physicsClientId=self.cid)
            lo, hi = map(np.array, self.p.getAABB(uid, j, physicsClientId=self.cid))
            pos0 = np.array(self.p.getLinkState(uid, j, computeForwardKinematics=1, physicsClientId=self.cid)[4])
            self.p.resetJointState(uid, j, 0.01, physicsClientId=self.cid)
            pos1 = np.array(self.p.getLinkState(uid, j, computeForwardKinematics=1, physicsClientId=self.cid)[4])
            self.p.resetJointState(uid, j, q0, physicsClientId=self.cid)
            self.fix_center.append((lo + hi) / 2)
            self.fix_half.append((hi - lo) / 2)
            self.fix_axis.append((pos1 - pos0) / 0.01)
        self.fix_center, self.fix_half, self.fix_axis = map(np.array, (self.fix_center, self.fix_half, self.fix_axis))
        # block half extents (box collision shapes centred on the base frame; sizes differ between scenes) and the
        # place of each block in scene_obs (pos + euler xyz after the 6 joint / light states, in the scene's order)
        movable = {o.name: o for o in env.scene.movable_objects}
        self.block_state = {o.name: 6 + 6 * i for i, o in enumerate(env.scene.movable_objects)}
        self.block_half = []
        for name in BLOCKS:
            (shape,) = self.p.getCollisionShapeData(movable[name].uid, -1, physicsClientId=self.cid)
            assert shape[2] == self.p.GEOM_BOX and np.allclose(shape[5], 0), name
            self.block_half.append(np.array(shape[3]) / 2)
        self.block_half = np.array(self.block_half)
        # node descriptors (node_id, role, slot), as in spikewam/skeleton.py
        self.n_nodes = N_NODES
        self.node_id = np.arange(N_NODES, dtype=np.int64)
        self.role = np.array([ROLES["robot"]] * N_ROBOT + [ROLES[r] for r in BOX_NODES] * MAX_SLOTS, np.int64)
        self.slot = np.array([0] * N_ROBOT + [m + 1 for m in range(MAX_SLOTS) for _ in range(SLOT_SIZE)], np.int64)

    def robot(self, robot_obs):
        """(F, 11, 3) robot nodes: forward kinematics of the arm joints and the gripper opening in the copy."""
        robot_obs = np.atleast_2d(robot_obs)
        out = np.zeros((len(robot_obs), N_ROBOT, 3))
        for f, ro in enumerate(robot_obs):
            for j, q in zip(self.arm_joints, ro[7:14]):
                self.p.resetJointState(self.robot_uid, j, q, physicsClientId=self.cid)
            for j in self.grip_joints:
                self.p.resetJointState(self.robot_uid, j, ro[6] / 2, physicsClientId=self.cid)
            states = self.p.getLinkStates(self.robot_uid, self.robot_links, computeForwardKinematics=1,
                                          physicsClientId=self.cid)
            out[f] = [s[4] for s in states]
        return out

    def objects(self, scene_obs):
        """(F, 7, 6, 3) block and fixture nodes."""
        scene_obs = np.atleast_2d(scene_obs)
        blocks = []
        for b, name in enumerate(BLOCKS):
            st = scene_obs[:, self.block_state[name]:self.block_state[name] + 6]
            rot = Rotation.from_euler("xyz", st[:, 3:]).as_matrix()      # pybullet's euler convention
            blocks.append(box_nodes(st[:, :3], self.block_half[b], rot))
        fixtures = [box_nodes(self.fix_center[k] + scene_obs[:, FIXTURE_STATE[name], None] * self.fix_axis[k],
                              self.fix_half[k]) for k, name in enumerate(FIXTURES)]
        return np.stack(blocks + fixtures, 1)

    def __call__(self, robot_obs, scene_obs):
        """(F, 53, 3) skeletons (float32) of F frames (or (53, 3) for one frame)."""
        one = np.ndim(robot_obs) == 1
        r, o = self.robot(robot_obs), self.objects(scene_obs)
        s = np.concatenate([r, o.reshape(len(o), -1, 3)], 1).astype(np.float32)
        return s[0] if one else s
