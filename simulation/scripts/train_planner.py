from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.envs import PlannerBallEnv
from hrl_soccer.ppo import PPOConfig, train_ppo


def eval_goal_dist(model, episodes: int = 20) -> float:
    env = PlannerBallEnv(seed=987)
    dists = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        info = {"goal_dist": 10.0}
        while not done:
            action, _, _ = model.act(obs, deterministic=True)
            obs, _, done, info = env.step(action)
        dists.append(info["goal_dist"])
    return float(sum(dists) / max(1, len(dists)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    env = PlannerBallEnv(seed=args.seed)
    cfg = PPOConfig(total_steps=args.total_steps, seed=args.seed, rollout_steps=1024)
    train_ppo(env, cfg, ROOT / "checkpoints" / "planner_policy.pt", deterministic_eval=eval_goal_dist)


if __name__ == "__main__":
    main()
