from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_soccer.mujoco_go2 import Go2BezierKickEnv
from hrl_soccer.ppo import ActorCritic


def pin_base(env: Go2BezierKickEnv):
    env.data.qpos[:7] = [0.0, 0.0, 0.29, 1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:6] = 0.0
    env.mujoco.mj_forward(env.model, env.data)


def collect_dataset(episodes: int, seed: int):
    env = Go2BezierKickEnv(seed=seed)
    obs_buf = []
    act_buf = []
    for _ in range(episodes):
        obs = env.reset()
        obs = env.settle(80)
        pin_base(env)
        obs = env._obs()
        done = False
        while not done:
            action = env.expert_action()
            obs_buf.append(obs.copy())
            act_buf.append(action.copy())
            obs, _, done, _ = env.step(action)
            pin_base(env)
            obs = env._obs()
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.float32)


def evaluate(model: ActorCritic, episodes: int = 10) -> tuple[float, float, float]:
    env = Go2BezierKickEnv(seed=606)
    moves = []
    heights = []
    toe_max_zs = []
    for _ in range(episodes):
        obs = env.reset()
        obs = env.settle(80)
        pin_base(env)
        obs = env._obs()
        start = env._ball_xy().copy()
        done = False
        info = {"ball_xy": start, "base_height": 0.0}
        max_toe_z = float(env._toe()[2])
        while not done:
            action, _, _ = model.act(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            pin_base(env)
            obs = env._obs()
            max_toe_z = max(max_toe_z, float(env._toe()[2]))
        moves.append(float(np.linalg.norm(info["ball_xy"] - start)))
        heights.append(float(info["base_height"]))
        toe_max_zs.append(max_toe_z)
    return float(np.mean(moves)), float(np.mean(heights)), float(np.mean(toe_max_zs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=505)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "go2_bc_policy.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    obs, acts = collect_dataset(args.episodes, args.seed)
    model = ActorCritic(obs.shape[1], acts.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    obs_t = torch.as_tensor(obs)
    acts_t = torch.as_tensor(acts)
    phase_t = obs_t[:, -1]
    sample_w = torch.ones_like(phase_t)
    sample_w = sample_w + torch.where(phase_t < 0.52, torch.full_like(sample_w, 3.0), torch.zeros_like(sample_w))
    action_w = torch.ones(acts_t.shape[1], dtype=torch.float32)
    action_w[3:6] = torch.tensor([2.0, 6.0, 4.0], dtype=torch.float32)

    best_score = -1.0
    best_metrics = (0.0, 0.0, 0.0)
    best_state = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(obs_t.shape[0])
        losses = []
        for start in range(0, obs_t.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            pred, value = model(obs_t[idx])
            err = (pred - acts_t[idx]) ** 2 * action_w
            loss = torch.mean(torch.mean(err, dim=1) * sample_w[idx]) + 1e-4 * torch.mean(value * value)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            move, height, toe_z = evaluate(model, episodes=args.eval_episodes)
            # Save the checkpoint that both kicks the ball and visibly lifts the foot.
            score = move + 20.0 * min(toe_z, 0.20)
            if score > best_score:
                best_score = score
                best_metrics = (move, height, toe_z)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"epoch={epoch:03d} loss={np.mean(losses):.5f} "
                f"eval_ball_move={move:.3f} eval_base_h={height:.3f} eval_max_toe_z={toe_z:.3f}"
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "obs_dim": obs.shape[1], "act_dim": acts.shape[1]}, args.checkpoint)
    print(
        f"saved {args.checkpoint} best_eval_ball_move={best_metrics[0]:.3f} "
        f"best_eval_base_h={best_metrics[1]:.3f} best_eval_max_toe_z={best_metrics[2]:.3f}"
    )


if __name__ == "__main__":
    main()
