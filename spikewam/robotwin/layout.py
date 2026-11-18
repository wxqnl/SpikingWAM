"""RoboTwin 2.0 skeleton layout (aloha-agilex), shared by the collector (RoboTwin python) and training
(`spikewam.train --bench robotwin`). Coordinates are metres in the RoboTwin world frame (both arm bases are fixed in
it; the randomized setting moves the table height).

Robot (20 nodes, 10 per arm, left arm first): the origins of the arm's base link and links 1-6, the grip site (the
midpoint of the two finger links; RoboTwin's end pose sits at the link-6 origin, 8 cm behind the fingers) and the two
finger links (7, 8).

Objects: one slot per task object (spikewam.robotwin.rt_env.task_actors, at most MAX_SLOTS). Each node of a slot has
a fixed index inside the slot (so a node identity means the same thing in every episode, whichever model the episode
drew):
  0       center   the centre of the model's bounding box (rigid objects) or the root link (articulated ones),
  1-6     faces    the centres of the box faces +x, -x, +y, -y, +z, -z in the model frame (rigid objects),
  7-10    contact  the annotated contact (grasp) points, at most 4,
  11-12   functional  annotated functional points (e.g. a hammer head), at most 2,
  13-14   target   annotated target points (where something is placed), at most 2,
  15      orientation  the annotated orientation point,
  16-19   part     the origins of the moving links of an articulated object (lid, door, drawer), at most 4.
"""

ARMS = ("fl", "fr")                                   # aloha-agilex left and right arm
ARM_LINKS = ["base_link"] + [f"link{i}" for i in range(1, 7)]
FINGERS = ["link7", "link8"]
ARM_NODES = len(ARM_LINKS) + 1 + len(FINGERS)         # links, grip site, fingers
N_ROBOT = 2 * ARM_NODES
GRIP_NODES = (len(ARM_LINKS), ARM_NODES + len(ARM_LINKS))
MAX_SLOTS = 6
SLOT_SIZE = 20
N_NODE_IDS = N_ROBOT + MAX_SLOTS * SLOT_SIZE
SLOT_INDEX = {"center": 0, "face": 1, "contact": 7, "functional": 11, "target": 13, "orientation": 15, "part": 16}
MAX_POINTS = {"contact": 4, "functional": 2, "target": 2, "orientation": 1, "part": 4}
ROLES = {"robot": 0, "grip": 1, "finger": 2, "center": 3, "face": 4, "contact": 5, "functional": 6, "target": 7,
         "orientation": 8, "part": 9}
N_ROLES = len(ROLES)
