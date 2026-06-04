from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old.*")

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 128)):
        super().__init__()
        layers = []
        last = obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        self.v = nn.Linear(last, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        return torch.tanh(self.mu(h)), self.v(h).squeeze(-1)

    def dist(self, obs: torch.Tensor):
        mu, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std), value

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist, value = self.dist(obs_t)
        raw = dist.mean if deterministic else dist.rsample()
        action = torch.clamp(raw, -1.0, 1.0)
        logp = dist.log_prob(raw).sum(-1)
        return action.squeeze(0).cpu().numpy(), float(logp.item()), float(value.item())


@dataclass
class PPOConfig:
    total_steps: int = 100_000
    rollout_steps: int = 1024
    epochs: int = 6
    minibatch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 1


def train_ppo(env, cfg: PPOConfig, checkpoint: Path, deterministic_eval=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = ActorCritic(env.spec.obs_dim, env.spec.act_dim)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.pi_lr)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    obs = env.reset()
    global_step = 0
    update = 0

    while global_step < cfg.total_steps:
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        ep_returns = []
        ep_ret = 0.0
        for _ in range(cfg.rollout_steps):
            action, logp, value = model.act(obs)
            next_obs, reward, done, _ = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value)
            ep_ret += reward
            obs = next_obs
            global_step += 1
            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs = env.reset()

        with torch.no_grad():
            _, last_val = model.dist(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            last_val = float(last_val.item())

        adv, ret = generalized_advantage(np.array(rew_buf), np.array(done_buf), np.array(val_buf), last_val, cfg.gamma, cfg.lam)
        obs_t = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32)
        act_t = torch.as_tensor(np.asarray(act_buf), dtype=torch.float32)
        old_logp_t = torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_buf)
        for _ in range(cfg.epochs):
            order = torch.randperm(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = order[start : start + cfg.minibatch_size]
                dist, value = model.dist(obs_t[idx])
                logp = dist.log_prob(act_t[idx]).sum(-1)
                ratio = torch.exp(logp - old_logp_t[idx])
                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_t[idx]
                pi_loss = -torch.min(unclipped, clipped).mean()
                v_loss = torch.mean((value - ret_t[idx]) ** 2)
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optim.step()

        update += 1
        if update % 2 == 0 or global_step >= cfg.total_steps:
            torch.save({"model": model.state_dict(), "obs_dim": env.spec.obs_dim, "act_dim": env.spec.act_dim}, checkpoint)
            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            eval_msg = ""
            if deterministic_eval is not None:
                eval_msg = f" eval={deterministic_eval(model):.3f}"
            print(f"step={global_step} update={update} mean_return={mean_ret:.3f}{eval_msg}")
    return model


def generalized_advantage(rewards, dones, values, last_value, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    returns = adv + values
    return adv, returns.astype(np.float32)


def load_policy(path: Path) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu")
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
