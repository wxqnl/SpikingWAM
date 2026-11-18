"""Survey of the RoboTwin 2.0 tasks for the skeleton definition: per task, the task objects (attribute, model,
articulated or not, number of contact / functional / target points) and, once, the robot links.

  cd $RT_ROOT && $RT_PY $SW_ROOT/scripts/robotwin_survey.py [task ...]
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from spikewam.robotwin.rt_env import make_args, make_task, task_actors  # noqa: E402

TASKS = sorted(f[:-3] for f in os.listdir("envs") if f.endswith(".py") and not f.startswith("_") and f != "__init__.py")


def count(actor, key):
    v = actor.config.get(actor.POINTS[key])
    return 0 if v is None else len(v)


def main():
    tasks = sys.argv[1:] or TASKS
    out, links_done = {}, False
    for name in tasks:
        task = make_task(name)
        args = make_args(name, "demo_clean")
        args["need_plan"], args["save_data"], args["render_freq"] = True, False, 0
        try:
            task.setup_demo(now_ep_num=0, seed=0, **args)
            if not links_done:
                r = task.robot
                print("left links", [l.get_name() for l in r.left_entity.get_links()])
                print("left arm joints", r.left_arm_joints_name, "ee", r.left_ee_name, "gripper", r.left_gripper_name)
                print("right arm joints", r.right_arm_joints_name, "ee", r.right_ee_name)
                links_done = True
            objs = []
            for attr, a in task_actors(task):
                objs.append({"attr": attr, "name": a.get_name(), "art": type(a).__name__ == "ArticulationActor",
                             "contact": count(a, "contact"), "functional": count(a, "functional"),
                             "target": count(a, "target"), "orient": a.config.get(a.POINTS["orientation"]) is not None,
                             "extents": a.config.get("extents")})
            out[name] = objs
            print(name, json.dumps(objs), flush=True)
        except Exception:
            print(name, "ERROR", traceback.format_exc()[-400:], flush=True)
        finally:
            try:
                task.close_env()
            except Exception:
                pass
    with open(os.path.join(os.environ.get("SW_RUNS", "."), "robotwin_survey.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
