from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.mujoco_go2 import GO2_SCENE, Go2BezierKickEnv
from hrl_soccer.ppo import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "go2_control_policy.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--no-pin-base", action="store_true", help="disable kinematic torso hold used for action preview")
    parser.add_argument("--freeze-after-ball-move", type=float, default=0.25)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="sample from the policy instead of using mean action")
    args = parser.parse_args()

    model = load_policy(args.checkpoint)
    env = Go2BezierKickEnv(GO2_SCENE, seed=303)

    viewer_ctx = None
    viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data)
        viewer = viewer_ctx.__enter__()

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            if args.settle_steps > 0:
                obs = env.settle(args.settle_steps)
            ball_start = env._ball_xy().copy()
            recovery = False
            done = False
            ret = 0.0
            toe_errors = []
            info = {"goal_dist": 0.0, "base_height": 0.0}
            while not done:
                step_start = time.time()
                if recovery:
                    action = env.recovery_action()
                else:
                    action, _, _ = model.act(obs, deterministic=not args.stochastic)
                obs, reward, done, info = env.step(action)
                if float(((info["ball_xy"] - ball_start) ** 2).sum() ** 0.5) >= args.freeze_after_ball_move:
                    recovery = True
                if not args.no_pin_base:
                    env.data.qpos[:7] = [0.0, 0.0, 0.29, 1.0, 0.0, 0.0, 0.0]
                    env.data.qvel[:6] = 0.0
                    env.mujoco.mj_forward(env.model, env.data)
                    obs = env._obs()
                ret += reward
                toe_errors.append(info["toe_error"])
                if viewer is not None:
                    viewer.sync()
                    if not viewer.is_running():
                        return
                    sleep = env.model.opt.timestep * env.decimation - (time.time() - step_start)
                    if sleep > 0:
                        time.sleep(sleep)
            mean_err = sum(toe_errors) / max(1, len(toe_errors))
            print(
                f"episode={ep + 1:02d} return={ret:.2f} mean_toe_error={mean_err:.3f} "
                f"goal_dist={info['goal_dist']:.3f} ball_move={float(((info['ball_xy'] - ball_start) ** 2).sum() ** 0.5):.3f} "
                f"base_h={info['base_height']:.3f}"
            )
    finally:
        if viewer_ctx is not None:
            sys.stdout.flush()
            # Some MuJoCo/GL driver combinations segfault during passive-viewer
            # teardown under `conda run`. Exit after flushing metrics to avoid
            # losing the result because of viewer cleanup.
            os._exit(0)


if __name__ == "__main__":
    main()
