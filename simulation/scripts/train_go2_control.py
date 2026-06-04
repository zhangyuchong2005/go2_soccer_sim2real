from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.mujoco_go2 import Go2BezierKickEnv
from hrl_soccer.ppo import PPOConfig, train_ppo


def eval_toe_error(model, episodes: int = 3) -> float:
    env = Go2BezierKickEnv(seed=202)
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
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "go2_control_policy.pt")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    if args.checkpoint.exists() and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.checkpoint.with_suffix(f".{stamp}.bak.pt")
        shutil.copy2(args.checkpoint, backup)
        print(f"backed up existing checkpoint to {backup}")
    env = Go2BezierKickEnv(seed=args.seed)
    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        minibatch_size=min(256, args.rollout_steps),
        seed=args.seed,
    )
    train_ppo(env, cfg, args.checkpoint, deterministic_eval=eval_toe_error)


if __name__ == "__main__":
    main()
