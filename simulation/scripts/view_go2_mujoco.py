from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.mujoco_go2 import GO2_SCENE, Go2BezierKickEnv


def main():
    import mujoco.viewer

    env = Go2BezierKickEnv(GO2_SCENE)
    env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            env.step([0.0] * env.act_dim)
            viewer.sync()
            sleep = env.model.opt.timestep * env.decimation - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
