from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.mujoco_go2 import GO2_SCENE, Go2BezierKickEnv


def main():
    env = Go2BezierKickEnv(GO2_SCENE)
    obs = env.reset()
    print(f"scene={GO2_SCENE}")
    print(f"nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} obs_dim={env.obs_dim} obs_shape={obs.shape}")
    for i in range(10):
        obs, reward, done, info = env.step([0.0] * env.act_dim)
        print(
            f"step={i + 1:02d} reward={reward:.3f} toe_error={info['toe_error']:.3f} "
            f"base_h={info['base_height']:.3f} ball=({info['ball_xy'][0]:.3f},{info['ball_xy'][1]:.3f})"
        )
        if done:
            break


if __name__ == "__main__":
    main()
