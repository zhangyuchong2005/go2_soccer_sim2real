from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .bezier import bezier, default_kick_curve, sample_kick_curve


@dataclass
class EnvSpec:
    obs_dim: int
    act_dim: int
    act_limit: float
    horizon: int


class LowLevelTrackingEnv:
    """Reduced-order approximation of the paper's low-level motion controller.

    The policy commands 12 joint targets. A simple latent toe model maps the front-right
    leg joints to toe position and rewards tracking of randomized Bezier trajectories.
    """

    def __init__(self, seed: int = 1, horizon: int = 240):
        self.rng = np.random.default_rng(seed)
        self.spec = EnvSpec(obs_dim=40, act_dim=12, act_limit=1.0, horizon=horizon)
        self.horizon = horizon
        self.dt = 0.02
        self.nominal_q = np.array([0.0, 0.75, -1.45] * 4, dtype=np.float32)
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.phase = 0
        self.q = self.nominal_q + self.rng.normal(0.0, 0.04, 12).astype(np.float32)
        self.dq = np.zeros(12, dtype=np.float32)
        self.base_rpy = self.rng.normal(0.0, 0.015, 3).astype(np.float32)
        self.curve = sample_kick_curve(self.rng)
        return self._obs()

    def _toe_position(self) -> np.ndarray:
        hip, thigh, calf = self.q[:3]
        x = 0.22 + 0.16 * np.sin(thigh) + 0.12 * np.sin(thigh + calf)
        y = -0.16 + 0.07 * np.sin(hip)
        z = -0.20 - 0.16 * np.cos(thigh) - 0.12 * np.cos(thigh + calf)
        return np.array([x, y, z], dtype=np.float32)

    def _ref(self) -> np.ndarray:
        t = min(1.0, self.step_count / max(1, self.horizon - 1))
        if t < 0.20:
            self.phase = 0
            return default_kick_curve()[:, 0]
        if t < 0.42:
            self.phase = 1
            return bezier(self.curve[:, :3], (t - 0.20) / 0.22)
        if t < 0.74:
            self.phase = 2
            return bezier(self.curve, (t - 0.42) / 0.32)
        self.phase = 3
        return bezier(self.curve[:, 2:], (t - 0.74) / 0.26)

    def _obs(self) -> np.ndarray:
        ref = self._ref()
        toe = self._toe_position()
        phase_onehot = np.eye(4, dtype=np.float32)[self.phase]
        return np.concatenate(
            [
                self.q,
                self.dq,
                self.base_rpy,
                toe,
                ref,
                ref - toe,
                phase_onehot,
            ]
        ).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_q = self.nominal_q + 0.65 * action
        kp = self.rng.uniform(0.13, 0.19)
        damping = self.rng.uniform(0.78, 0.88)
        self.dq = damping * self.dq + kp * (target_q - self.q)
        self.q = self.q + self.dq
        self.base_rpy = 0.985 * self.base_rpy + 0.006 * np.array(
            [self.q[0] - self.nominal_q[0], self.q[4] - self.nominal_q[4], self.q[8] - self.nominal_q[8]],
            dtype=np.float32,
        )

        self.step_count += 1
        toe = self._toe_position()
        ref = self._ref()
        err = float(np.linalg.norm(toe - ref))
        stability = float(np.linalg.norm(self.base_rpy))
        reward = np.exp(-18.0 * err * err) + 0.15 * np.exp(-10.0 * stability * stability) - 0.002 * float(np.dot(action, action))
        fallen = stability > 0.55 or err > 0.75
        done = fallen or self.step_count >= self.horizon
        if fallen:
            reward -= 1.0
        return self._obs(), float(reward), bool(done), {"toe_error": err, "stability": stability}


class PlannerBallEnv:
    """High-level soccer shooting planner.

    The action is a flattened 3x5 Bezier parameter table. The low-level controller is
    represented by a calibrated impulse model, with randomized ball and ground dynamics.
    """

    def __init__(self, seed: int = 7, horizon: int = 80, rho_goal: float = 1.8):
        self.rng = np.random.default_rng(seed)
        self.spec = EnvSpec(obs_dim=153, act_dim=15, act_limit=1.0, horizon=horizon)
        self.horizon = horizon
        self.rho_goal = rho_goal
        self.history = deque(maxlen=15)
        self.robot_hist = deque(maxlen=6)
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.goal = self.rng.uniform([1.4, -0.9], [3.0, 0.9]).astype(np.float32)
        self.ball = self.rng.uniform([0.35, -0.25], [0.75, 0.25]).astype(np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(15, dtype=np.float32)
        self.phase = 0
        self.mass_scale = float(self.rng.uniform(0.75, 1.35))
        self.friction = float(self.rng.uniform(0.935, 0.985))
        self.sensor_noise = float(self.rng.uniform(0.0, 0.025))
        self.delay = int(self.rng.integers(0, 3))
        self.history.clear()
        self.robot_hist.clear()
        for _ in range(15):
            self.history.append(self._noisy_ball3())
        for _ in range(6):
            self.robot_hist.append(np.zeros(15, dtype=np.float32))
        return self._obs()

    def _noisy_ball3(self) -> np.ndarray:
        xy = self.ball + self.rng.normal(0.0, self.sensor_noise, 2)
        return np.array([xy[0], xy[1], 0.105], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        ball_hist = list(self.history)
        if self.delay:
            ball_hist = [ball_hist[max(0, i - self.delay)] for i in range(len(ball_hist))]
        return np.concatenate(
            [
                self.goal,
                np.asarray(ball_hist, dtype=np.float32).reshape(-1),
                self.prev_action,
                np.asarray(list(self.robot_hist), dtype=np.float32).reshape(-1),
                np.array([self.phase / 3.0], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _curve_from_action(self, action: np.ndarray) -> np.ndarray:
        delta = np.clip(action.reshape(3, 5), -1.0, 1.0)
        base = default_kick_curve()
        scale = np.array([[0.32], [0.34], [0.18]], dtype=np.float32)
        curve = base + scale * delta
        curve[0] = np.clip(curve[0], 0.05, 0.95)
        curve[1] = np.clip(curve[1], -0.55, 0.55)
        curve[2] = np.clip(curve[2], -0.50, -0.01)
        return curve.astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_action = self.prev_action.copy()
        curve = self._curve_from_action(action)
        contact = curve[:, -1][:2]
        approach = curve[:, -1][:2] - curve[:, 2][:2]
        aim = self.goal - self.ball
        aim_norm = np.linalg.norm(aim) + 1e-6
        aim_dir = aim / aim_norm
        contact_err = np.linalg.norm(contact - self.ball)
        direction_quality = float(np.dot(approach / (np.linalg.norm(approach) + 1e-6), aim_dir))
        speed = max(0.0, direction_quality) * np.linalg.norm(approach) * np.exp(-12.0 * contact_err * contact_err)
        impulse = (1.9 / self.mass_scale) * speed * aim_dir

        # Ball rolls after the kick; this approximates the planner's interaction loop.
        self.ball_vel += impulse.astype(np.float32)
        for _ in range(8):
            self.ball += 0.08 * self.ball_vel
            self.ball_vel *= self.friction

        self.step_count += 1
        self.phase = min(3, self.step_count // max(1, self.horizon // 4))
        self.prev_action = action.astype(np.float32)
        self.history.append(self._noisy_ball3())
        self.robot_hist.append(np.concatenate([curve[:, -1], curve[:, 2], action[:9]]).astype(np.float32))

        dist = float(np.linalg.norm(self.ball - self.goal))
        reached = dist <= 0.2
        reward = 1.0 if reached else float(np.exp(-self.rho_goal * dist * dist))
        reward -= 0.01 * float(np.mean(np.square(action - prev_action)))
        stopped_far = np.linalg.norm(self.ball_vel) < 0.015 and self.step_count > 8 and not reached
        done = reached or stopped_far or self.step_count >= self.horizon
        return self._obs(), reward, bool(done), {"goal_dist": dist, "reached": reached}
