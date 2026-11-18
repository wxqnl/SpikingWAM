"""One closed-loop RoboTwin 2.0 episode of the final policy with the perception front end, keeping what the figure needs at
every replanning step: the native head-camera image, its world-to-pixel matrix, the skeleton the policy read (front-end
object nodes, joint arm nodes) and the simulator skeleton. Scene, instruction, policy and front-end settings are those
of the 100-episode evaluation (evaluate_seeds.py, eval_unit.sh: k 3, n_exec 16, PERCEPT_RIGID=1); the episode is one
seed of a finished evaluation's result file (--index, default the first seed that succeeded there). RoboTwin python,
cwd = the RoboTwin root:

  cd $RT_ROOT && PERCEPT_RIGID=1 $RT_PY $SW_ROOT/scripts/rollout_frames_rt.py --task pick_dual_bottles \
      --result $SW_ROOT/runs/robotwin_eval/rt_f2ppt16_big_percept_rtB30tCM_rigid/demo_randomized/pick_dual_bottles.json \
      --perception $SW_ROOT/runs/perception/rtB30tCM/ckpt.pt --out $SW_ROOT/runs/fig_rollouts/rt_pick_dual_bottles.npz
"""

import argparse
import json
import os
import sys

SW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(SW, "spikewam", "robotwin"))

import numpy as np  # noqa: E402

from evaluate import make_args, make_task, rollout, SkeletonPolicy  # noqa: E402  (evaluate.py sets up the paths)
from evaluate_seeds import EXPERT_FIRST  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", default="demo_randomized")
    p.add_argument("--result", required=True, help="result file of a finished evaluation (seeds, instructions, outcomes)")
    p.add_argument("--index", type=int, default=-1, help="episode of the result file (-1: the first that succeeded)")
    p.add_argument("--ckpt", default=os.path.join(SW, "runs/rt_f2ppt16_big/ckpt_150000.pt"))
    p.add_argument("--perception", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ref = json.load(open(a.result))
    outcomes = [r["success"] for r in ref["results"][list(ref["results"])[0]]]
    i = a.index if a.index >= 0 else outcomes.index(True)
    seed, instruction = ref["seeds"][i], ref["instructions"][i]

    policy = SkeletonPolicy(a.ckpt, k=3, h=10, n_exec=16, seed=0, consensus="medoid", flow_steps=0, fixed_noise=False,
                            temporal=0.0, grip_bin=False)
    from spikewam.robotwin.percept import RTFrontEnd
    percept = RTFrontEnd(a.perception, policy.device)
    keep = {k: [] for k in ("head", "head_M", "mats", "skel", "skel_sim")}
    call = percept.__call__

    def recording(task, sk, skel, text):
        est = call(task, sk, skel, text)
        cfg = task.cameras.get_config()["head_camera"]
        keep["head"].append(task.cameras.get_rgb()["head_camera"]["rgb"][..., :3].copy())
        keep["head_M"].append(np.asarray(cfg["intrinsic_cv"]) @ np.asarray(cfg["extrinsic_cv"])[:3])
        keep["mats"].append(percept.last_views[1])
        keep["skel"].append(est)
        keep["skel_sim"].append(skel.copy())
        return est
    percept.last_cm = None

    class Wrap:                                  # rollout calls percept(task, sk, skel, text) and reads .last_cm
        def __call__(self, *args):
            return recording(*args)

        @property
        def last_cm(self):
            return percept.last_cm

    task = make_task(a.task)
    args = make_args(a.task, a.config)
    args.update(render_freq=0, eval_mode=True, eval_video_log=False, policy_name="SpikeWAM", ckpt_setting="sw")
    if a.task in EXPERT_FIRST:
        task.setup_demo(now_ep_num=i, seed=seed, is_test=True, **args)
        task.play_once()
        task.close_env()
    task.setup_demo(now_ep_num=i, seed=seed, is_test=True, **args)
    task.set_instruction(instruction=instruction)
    rec = {"q": [], "skel": [], "act": []}
    succ, steps, plans, cms = rollout(task, policy, instruction, 16, rec, -1.0, seed, Wrap())
    task.close_env()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in keep.items()}, success=succ, steps=steps, seed=seed,
                        index=i, instruction=instruction, role=rec["role"], slot=rec["slot"],
                        objects=np.asarray(rec["objects"]), cm=np.array(cms), front_end=a.perception)
    print(f"{a.task} seed {seed} (episode {i}): success {succ} (evaluation: {outcomes[i]}), {steps} steps, {plans} plans, "
          f"front-end error {np.mean(cms):.2f} cm")


if __name__ == "__main__":
    main()
