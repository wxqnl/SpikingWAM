"""CALVIN skeleton layout (Franka Panda, play table A/B/C/D), shared by the extractor (scripts/calvin_extract.py),
training (`spikewam.train --bench calvin`) and the evaluator (spikewam/calvin/evaluate.py). Coordinates are metres in
the CALVIN world frame (robot base and table are fixed in it; the table and its fixtures sit elsewhere in each scene).

Robot (11 nodes, as in LIBERO, spikewam/skeleton.py): the frame origins of Panda links 1-7 and of the hand, the TCP
(the grip site) and the two finger tips (they carry the gripper opening).

Objects: one slot per scene object, always all seven, in a fixed order: the red, blue and pink block, the sliding door,
the drawer, the button and the switch. Each slot is a box of six nodes,
  center, top (+z face), +x, -x, +y, -y face centres,
in the block frame for the blocks (their sizes differ between scenes), and on the world-aligned box of the link for the
four fixtures (all four are prismatic, so the box only translates with the joint). The lightbulb and the LED are
lights: their state follows the switch position and the button presses, and they never move. scene_obs lists the
blocks in the order of each scene's configuration (CalvinSkeleton.block_state), the fixtures in the same order in all
four scenes.
"""

ROBOT_LINKS = ["panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6",
               "panda_link7", "panda_hand", "tcp", "finger_left_tip", "finger_right_tip"]
N_ROBOT = len(ROBOT_LINKS)
GRIP_NODES = (8,)
BLOCKS = ("block_red", "block_blue", "block_pink")
FIXTURES = ("slide_link", "drawer_link", "button_link", "switch_link")     # table links
FIXTURE_STATE = {"slide_link": 0, "drawer_link": 1, "button_link": 2, "switch_link": 3}   # index in scene_obs
SLOTS = BLOCKS + FIXTURES
MAX_SLOTS = len(SLOTS)
BOX_NODES = ("center", "top", "side+x", "side-x", "side+y", "side-y")
SLOT_SIZE = len(BOX_NODES)
N_NODES = N_ROBOT + MAX_SLOTS * SLOT_SIZE
N_NODE_IDS = N_NODES
ROLES = {"robot": 0, "center": 1, "top": 2, "side+x": 3, "side-x": 4, "side+y": 5, "side-y": 6}
N_ROLES = len(ROLES)
