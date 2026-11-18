"""Sim-state skeleton of a RoboTwin 2.0 scene (RoboTwin python; layout in spikewam.robotwin.layout).

Built once per episode after the task has loaded its objects (`RTSkeleton(task)`); calling it returns the
(n_nodes, 3) float32 node positions for the simulator's current state.
"""

from __future__ import annotations

import numpy as np

from .layout import ARM_LINKS, ARMS, FINGERS, MAX_POINTS, MAX_SLOTS, N_ROBOT, ROLES, SLOT_INDEX, SLOT_SIZE
from .rt_env import task_actors


def _box(cfg):
    """Box centre and half extents in the actor frame. Primitives (create_box / create_cylinder) store the half size
    in both `extents` and `scale` (their points live on the unit box); meshes store full extents in model units."""
    scale = np.asarray(cfg.get("scale", 1.0), float) * np.ones(3)
    if "extents" not in cfg:
        return None
    ext = np.asarray(cfg["extents"], float)
    if np.allclose(ext, scale):
        return np.zeros(3), scale
    return np.asarray(cfg.get("center", [0, 0, 0]), float) * scale, ext * scale / 2


class RTSkeleton:
    def __init__(self, task):
        self.task = task
        robot = task.robot
        self.arm_links = []
        for arm, ent in zip(ARMS, (robot.left_entity, robot.right_entity)):
            links = {l.get_name(): l for l in ent.get_links()}
            self.arm_links.append(([links[f"{arm}_{n}"] for n in ARM_LINKS], [links[f"{arm}_{n}"] for n in FINGERS]))
        objects = task_actors(task)
        assert len(objects) <= MAX_SLOTS, [n for n, _ in objects]
        self.object_names = [n for n, _ in objects]
        node_id = list(range(N_ROBOT))
        role = []
        for _ in ARMS:
            role += [ROLES["robot"]] * len(ARM_LINKS) + [ROLES["grip"]] + [ROLES["finger"]] * len(FINGERS)
        slot = [0] * N_ROBOT
        local = [np.full(3, np.nan)] * N_ROBOT
        self.getters = []                       # per object: list of functions actor -> (3,) world position
        for m, (_, actor) in enumerate(objects):
            items = self._object_nodes(actor)
            base = N_ROBOT + m * SLOT_SIZE
            node_id += [base + it[0] for it in items]
            role += [ROLES[it[1]] for it in items]
            slot += [m + 1] * len(items)
            local += [np.full(3, np.nan) if it[3] is None else np.asarray(it[3], float)[:3] for it in items]
            self.getters.append((actor, [it[2] for it in items]))
        self.node_id = np.asarray(node_id, np.int64)
        self.role = np.asarray(role, np.int64)
        self.slot = np.asarray(slot, np.int64)
        self.local = np.stack(local)            # model-frame point of each node of a rigid object (from its model), NaN otherwise
        self.n_nodes = len(node_id)

    @staticmethod
    def _object_nodes(actor):
        """[(index in slot, role, f(actor) -> world position, model-frame point or None)] for one object."""
        cfg = actor.config
        art = type(actor).__name__ == "ArticulationActor"
        items = []
        if art:
            items.append((SLOT_INDEX["center"], "center", lambda a: np.asarray(a.actor.get_root_pose().p), None))
        else:
            box = _box(cfg)
            if box is None:
                items.append((SLOT_INDEX["center"], "center", lambda a: np.asarray(a.get_pose().p), np.zeros(3)))
            else:
                c, h = box
                local = [c] + [c + s * h[k] * np.eye(3)[k] for k in range(3) for s in (1, -1)]
                for i, p in enumerate(local):
                    idx = SLOT_INDEX["center"] if i == 0 else SLOT_INDEX["face"] + i - 1
                    items.append((idx, "center" if i == 0 else "face",
                                  lambda a, p=p: a.get_pose().to_transformation_matrix()[:3] @ np.append(p, 1.0), p))
        scale = np.asarray(cfg.get("scale", 1.0), float)
        for kind in ("contact", "functional", "target"):
            pts = cfg.get(actor.POINTS[kind]) or []
            for i in range(min(len(pts), MAX_POINTS[kind])):
                # Actor.get_point scales in place and fails on integer annotation matrices; same transform, in float
                if art:
                    p = np.append(np.asarray(pts[i]["matrix"], float)[:3, 3] * scale, 1.0)
                    link = actor.link_dict[pts[i]["base"]]
                    f = lambda a, p=p, link=link: link.get_pose().to_transformation_matrix()[:3] @ p
                else:
                    p = np.append(np.asarray(pts[i], float)[:3, 3] * scale, 1.0)
                    f = lambda a, p=p: a.get_pose().to_transformation_matrix()[:3] @ p
                items.append((SLOT_INDEX[kind] + i, kind, f, None if art else p))
        orient = np.asarray(cfg.get(actor.POINTS["orientation"]) or [], float)
        if not art and orient.shape == (4, 4):      # one matrix (Actor.get_orientation_point indexes its first row)
            p = np.append(orient[:3, 3] * np.asarray(cfg.get("scale", 1.0), float), 1.0)
            items.append((SLOT_INDEX["orientation"], "orientation",
                          lambda a, p=p: a.get_pose().to_transformation_matrix()[:3] @ p, p))
        if art:
            links = actor.actor.get_links()[1:1 + MAX_POINTS["part"]]
            for i, link in enumerate(links):
                items.append((SLOT_INDEX["part"] + i, "part", lambda a, link=link: np.asarray(link.get_pose().p), None))
        return items

    def __call__(self):
        pts = []
        for links, fingers in self.arm_links:
            pts += [np.asarray(l.get_pose().p) for l in links]
            tips = [np.asarray(l.get_pose().p) for l in fingers]
            pts += [(tips[0] + tips[1]) / 2] + tips
        for actor, fs in self.getters:
            pts += [f(actor) for f in fs]
        return np.stack(pts).astype(np.float32)
