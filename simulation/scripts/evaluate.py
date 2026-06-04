from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.envs import LowLevelTrackingEnv, PlannerBallEnv
from hrl_soccer.ppo import load_policy


def eval_control(path: Path):
    model = load_policy(path)
    env = LowLevelTrackingEnv(seed=77)
    errors = []
    returns = []
    for _ in range(8):
        obs = env.reset()
        done = False
        ret = 0.0
        while not done:
            action, _, _ = model.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ret += reward
            errors.append(info["toe_error"])
        returns.append(ret)
    print(f"control mean_return={sum(returns)/len(returns):.3f} mean_toe_error={sum(errors)/len(errors):.3f} m")


def eval_planner(path: Path):
    model = load_policy(path)
    env = PlannerBallEnv(seed=88)
    dists = []
    reached = 0
    for _ in range(30):
        obs = env.reset()
        done = False
        info = {"goal_dist": 99.0, "reached": False}
        while not done:
            action, _, _ = model.act(obs, deterministic=True)
            obs, _, done, info = env.step(action)
        dists.append(info["goal_dist"])
        reached += int(info["reached"])
    print(f"planner success={reached}/30 mean_goal_dist={sum(dists)/len(dists):.3f} m")


def main():
    control = ROOT / "checkpoints" / "control_policy.pt"
    planner = ROOT / "checkpoints" / "planner_policy.pt"
    if control.exists():
        eval_control(control)
    else:
        print("control checkpoint missing; run scripts/train_control.py first")
    if planner.exists():
        eval_planner(planner)
    else:
        print("planner checkpoint missing; run scripts/train_planner.py first")


if __name__ == "__main__":
    main()
