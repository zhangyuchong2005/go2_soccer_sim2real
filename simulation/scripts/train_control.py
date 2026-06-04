from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.envs import LowLevelTrackingEnv
from hrl_soccer.ppo import PPOConfig, train_ppo


def eval_error(model, episodes: int = 5) -> float:
    env = LowLevelTrackingEnv(seed=123)
    errors = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _, _ = model.act(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            errors.append(info["toe_error"])
    return float(sum(errors) / max(1, len(errors)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    env = LowLevelTrackingEnv(seed=args.seed)
    cfg = PPOConfig(total_steps=args.total_steps, seed=args.seed)
    train_ppo(env, cfg, ROOT / "checkpoints" / "control_policy.pt", deterministic_eval=eval_error)


if __name__ == "__main__":
    main()
