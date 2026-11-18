"""Sim-state scene skeleton for LIBERO (SkeleWAM, arXiv 2610.02120, Sec. 3.2).

SkeleWAM does not release its node definition, so this module fixes ours. All coordinates are metres in the robot
base frame.

Robot (11 nodes, from the simulator, i.e. forward kinematics): the origins of Panda links 1-7 and of the hand, the
grip site, and the two finger tips (the tips carry the gripper opening).

Objects: one group per entry of the task's BDDL `obj_of_interest` list, in that order (at most 4 groups). An entry
names an object, a site of an object (e.g. `wooden_cabinet_1_middle_region`), or a table region. A group holds
  center   the centre of the object's axis-aligned box in its own root frame (all of its geoms, joints at zero),
  top      the centre of that box's top face,
  4 sides  the centres of its +x, -x, +y, -y faces at mid height (they turn with the object),
  sites    the object's named sites except `default_site`; the entry's own site comes first,
  parts    the box centre of every moving part (drawers, doors, knobs; bodies other than `main` and `base`).
A table region (e.g. `main_table_stove_front_region`) is a group of one node at the site.

Each node carries three integer descriptors for the model's identity embeddings: `node_id` (robot node j -> j,
object node i of slot m -> N_ROBOT + m * SLOT_SIZE + i), `role` (ROLES) and `slot` (0 robot, 1..4 object slots).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import re
import numpy as np

try:                 # the node constants are also used where MuJoCo is not installed (RoboTwin's python)
    import mujoco
except ImportError:
    mujoco = None

ROBOT_BODIES = [f"robot0_link{i}" for i in range(1, 8)] + ["robot0_right_hand"]
ROBOT_SITES = ["gripper0_grip_site"]
ROBOT_TIPS = ["gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip"]
N_ROBOT = len(ROBOT_BODIES) + len(ROBOT_SITES) + len(ROBOT_TIPS)
MAX_SLOTS = 4
SLOT_SIZE = 16
N_NODE_IDS = N_ROBOT + MAX_SLOTS * SLOT_SIZE
ROLES = {"robot": 0, "center": 1, "top": 2, "side+x": 3, "side-x": 4, "side+y": 5, "side-y": 6,
         "focus": 7, "site": 8, "region": 9, "part": 10}
N_ROLES = len(ROLES)
# Robot edges follow the kinematic chain: link1..link7, hand, grip site; both tips hang off the hand.
ROBOT_EDGES = [(i, i + 1) for i in range(len(ROBOT_BODIES) - 1)] + [(7, 8), (7, 9), (7, 10)]


def _geom_points(model, data, g):
    """Points that bound geom g, in world coordinates (mesh vertices or primitive box corners)."""
    gtype = model.geom_type[g]
    size = model.geom_size[g]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        local = model.mesh_vert[adr:adr + num]
    elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.zeros((0, 3))
    else:
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.array([size[0]] * 3)
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = np.array([size[0], size[0], size[1] + size[0]])
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = np.array([size[0], size[0], size[1]])
        else:  # box, ellipsoid
            half = np.asarray(size[:3])
        local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    return local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]


def _local_box(model, data, body_ids, frame_body):
    """Axis-aligned box (centre, half extents) of the geoms of body_ids, in the frame of frame_body."""
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in body_ids]
    pts = [_geom_points(model, data, g) for g in geoms]
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(pts) == 0:
        return np.zeros(3), np.zeros(3)
    rot, pos = data.xmat[frame_body].reshape(3, 3), data.xpos[frame_body]
    local = (pts - pos) @ rot
    lo, hi = local.min(0), local.max(0)
    return (lo + hi) / 2, (hi - lo) / 2


@dataclass
class _Group:
    root: int = -1                     # root body id (-1 for a table region)
    center: np.ndarray = None          # box centre in the root frame
    half: np.ndarray = None            # box half extents in the root frame
    sites: list = field(default_factory=list)   # (site id, role)
    parts: list = field(default_factory=list)   # (body id, box centre in the part frame)


class SkeletonExtractor:
    """Builds the skeleton of one LIBERO scene. Create it once per environment (model), call it per state."""

    def __init__(self, sim, object_names, obj_of_interest):
        model = sim.model._model
        self.model = model
        self.base = model.body("robot0_base").id
        self.robot_bodies = [model.body(n).id for n in ROBOT_BODIES]
        self.robot_sites = [model.site(n).id for n in ROBOT_SITES]
        self.robot_tips = [model.body(n).id for n in ROBOT_TIPS]
        assert len(obj_of_interest) <= MAX_SLOTS, obj_of_interest
        # Object boxes at the model's reference configuration (joints at zero: drawers closed, doors shut).
        ref = mujoco.MjData(model)
        mujoco.mj_forward(model, ref)
        body_names = [model.body(i).name for i in range(model.nbody)]
        site_names = [model.site(i).name for i in range(model.nsite)]
        self.groups = []
        node_id, role, slot, edges = [], [], [], list(ROBOT_EDGES)
        node_id += list(range(N_ROBOT))
        role += [ROLES["robot"]] * N_ROBOT
        slot += [0] * N_ROBOT
        for m, entry in enumerate(obj_of_interest):
            obj, focus = self._resolve(entry, object_names)
            grp = _Group()
            if obj is None:  # a table region: one node at the site
                grp.sites = [(model.site(entry).id, ROLES["region"])]
                roles = [ROLES["region"]]
            else:
                bodies = [i for i, n in enumerate(body_names) if n.startswith(obj + "_")]
                grp.root = model.body(obj + "_main").id
                grp.center, grp.half = _local_box(model, ref, set(bodies), grp.root)
                sites = sorted(n for n in site_names if n.startswith(obj + "_") and n != obj + "_default_site")
                if focus is not None:
                    sites = [focus] + [s for s in sites if s != focus]
                for n in sites:
                    r = ROLES["focus"] if n == focus else (ROLES["region"] if n.endswith("region") else ROLES["site"])
                    grp.sites.append((model.site(n).id, r))
                for b in sorted(bodies, key=lambda i: body_names[i]):
                    if body_names[b] in (obj + "_main", obj + "_base"):
                        continue
                    c, _ = _local_box(model, ref, {b}, b)
                    grp.parts.append((b, c))
                roles = [ROLES[k] for k in ("center", "top", "side+x", "side-x", "side+y", "side-y")]
                roles += [r for _, r in grp.sites] + [ROLES["part"]] * len(grp.parts)
            assert len(roles) <= SLOT_SIZE, (entry, len(roles))
            first = len(node_id)
            self.kinds = getattr(self, "kinds", []) + [re.sub(r"_\d+(?=_|$)", "", entry)] * len(roles)   # object nodes only
            edges += [(first, first + i) for i in range(1, len(roles))]
            node_id += [N_ROBOT + m * SLOT_SIZE + i for i in range(len(roles))]
            role += roles
            slot += [m + 1] * len(roles)
            self.groups.append(grp)
        self.node_id = np.asarray(node_id, np.int64)
        self.role = np.asarray(role, np.int64)
        self.slot = np.asarray(slot, np.int64)
        self.edges = np.asarray(edges, np.int64)
        self.n_nodes = len(node_id)

    @staticmethod
    def _resolve(entry, object_names):
        """Map an obj_of_interest entry to (object name or None, focus site name or None)."""
        if entry in object_names:
            return entry, None
        for n in sorted(object_names, key=len, reverse=True):
            if entry.startswith(n + "_"):
                return n, entry
        return None, entry

    def world_points(self, data):
        """(n_nodes, 3) node positions in the world frame for MjData `data`."""
        pts = [data.xpos[b] for b in self.robot_bodies]
        pts += [data.site_xpos[s] for s in self.robot_sites]
        pts += [data.xpos[b] for b in self.robot_tips]
        for grp in self.groups:
            if grp.root >= 0:
                rot, pos = data.xmat[grp.root].reshape(3, 3), data.xpos[grp.root]
                c, h = grp.center, grp.half
                local = [c, c + [0, 0, h[2]], c + [h[0], 0, 0], c - [h[0], 0, 0], c + [0, h[1], 0], c - [0, h[1], 0]]
                pts += [rot @ p + pos for p in local]
            pts += [data.site_xpos[s] for s, _ in grp.sites]
            for b, c in grp.parts:
                pts.append(data.xmat[b].reshape(3, 3) @ c + data.xpos[b])
        return np.stack(pts)

    def __call__(self, sim):
        """(n_nodes, 3) float32 skeleton in the robot base frame for the simulator's current state."""
        data = sim.data._data
        rot, pos = data.xmat[self.base].reshape(3, 3), data.xpos[self.base]
        return ((self.world_points(data) - pos) @ rot).astype(np.float32)
