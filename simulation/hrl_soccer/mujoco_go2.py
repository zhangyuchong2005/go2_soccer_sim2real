from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .envs import EnvSpec


ROOT = Path(__file__).resolve().parents[1]
GO2_SCENE = ROOT / "assets" / "go2_soccer_scene.xml"


GO2_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


@dataclass
class Go2StepInfo:
    toe_error: float
    base_height: float
    ball_xy: np.ndarray
    goal_dist: float


class Go2BezierKickEnv:
    """MuJoCo Go2 full-body kicking environment.

    The action is a 12D residual joint-position command. It is converted to torques
    through PD control and stepped in MuJoCo. The front-right foot tracks a sampled
    Bezier toe trajectory near the soccer ball.
    """

    def __init__(
        self,
        xml_path: str | Path = GO2_SCENE,
        seed: int = 11,
        sim_dt: float = 0.002,
        control_dt: float = 0.02,
        horizon: int = 180,
    ):
        import mujoco

        self.mujoco = mujoco
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        self.decimation = max(1, int(round(control_dt / sim_dt)))
        self.horizon = horizon
        support_q = [0.0, 0.65, -1.45]
        swing_q = [0.0, 0.9, -1.8]
        self.default_q = np.array(support_q + swing_q + support_q + support_q, dtype=np.float32)
        self.kp = np.array([35.0, 45.0, 45.0] * 4, dtype=np.float32)
        self.kd = np.array([0.9, 1.2, 1.2] * 4, dtype=np.float32)
        self.action_scale = np.array([0.35, 0.55, 0.65] * 4, dtype=np.float32)
        self.action_scale[3:6] = np.array([0.60, 1.40, 1.05], dtype=np.float32)
        self.support_mask = np.ones(12, dtype=bool)
        self.support_mask[3:6] = False
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in GO2_JOINT_ORDER]
        self.fr_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "FR")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.goal_xy = np.array([2.3, 0.0], dtype=np.float32)
        self.obs_dim = 12 + 12 + 3 + 3 + 3 + 2 + 1
        self.act_dim = 12
        self.spec = EnvSpec(obs_dim=self.obs_dim, act_dim=self.act_dim, act_limit=1.0, horizon=horizon)
        self.action_filter = 0.25
        self.balance_assist = False
        self.base_height_target = 0.29
        self.total_mass = float(np.sum(self.model.body_mass))
        self.reset()

    def reset(self) -> np.ndarray:
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.45], dtype=np.float64)
        self.data.qpos[7:19] = self.default_q
        ball_xy = self.rng.uniform([0.370, -0.160], [0.395, -0.145])
        ball_qpos = self.model.jnt_qposadr[self.model.body_jntadr[self.ball_body_id]]
        self.data.qpos[ball_qpos : ball_qpos + 3] = [ball_xy[0], ball_xy[1], 0.105]
        self.data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.step_count = 0
        self.last_action = np.zeros(12, dtype=np.float32)
        self.mujoco.mj_forward(self.model, self.data)
        self.prev_goal_dist = float(np.linalg.norm(self._ball_xy() - self.goal_xy))
        self.start_toe = self._toe().copy()
        return self._obs()

    def _q_dq(self):
        return self.data.qpos[7:19].copy(), self.data.qvel[6:18].copy()

    def _toe(self) -> np.ndarray:
        return self.data.geom_xpos[self.fr_geom_id].copy().astype(np.float32)

    def _ball_xy(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id, :2].copy().astype(np.float32)

    def _ref(self) -> np.ndarray:
        t = min(1.0, self.step_count / max(1, self.horizon - 1))
        ball = self._ball_xy()
        aim = self.goal_xy - ball
        aim_dir = aim / (np.linalg.norm(aim) + 1e-6)
        back = np.array([-aim_dir[0], -aim_dir[1], 0.0], dtype=np.float32)
        impact = np.array([ball[0], ball[1], 0.105], dtype=np.float32) + 0.12 * back
        follow = np.array([ball[0], ball[1], 0.115], dtype=np.float32) + 0.18 * np.r_[aim_dir, 0.0].astype(np.float32)
        lift = self.start_toe + np.array([0.08, -0.03, 0.24], dtype=np.float32)
        prep = impact + np.array([-0.20 * aim_dir[0], -0.20 * aim_dir[1], 0.14], dtype=np.float32)
        if t < 0.25:
            u = t / 0.25
            return (1.0 - u) * self.start_toe + u * lift
        if t < 0.60:
            u = (t - 0.25) / 0.35
            return (1.0 - u) * lift + u * prep
        if t < 0.82:
            u = (t - 0.60) / 0.22
            return (1.0 - u) * prep + u * impact
        u = (t - 0.82) / 0.18
        return (1.0 - u) * impact + u * follow

    def _obs(self) -> np.ndarray:
        q, dq = self._q_dq()
        toe = self._toe()
        ref = self._ref()
        return np.concatenate(
            [
                q.astype(np.float32),
                dq.astype(np.float32),
                self.data.qpos[:3].astype(np.float32),
                toe,
                ref,
                self._ball_xy(),
                np.array([self.step_count / self.horizon], dtype=np.float32),
            ]
        )

    def expert_action(self) -> np.ndarray:
        phase = self.step_count / max(1, self.horizon - 1)
        action = np.zeros(12, dtype=np.float32)
        # FR leg swing primitive in normalized action space:
        # 1. Lift: raise FR leg high with backswing preparation
        # 2. Backswing: swing leg far backward above the ball
        # 3. Strike: sweep forward through the ball center line
        # 4. Follow-through: continue forward then recover
        lift = np.array([0.00, 0.65, -1.60], dtype=np.float32)
        backswing = np.array([0.00, 1.00, -1.80], dtype=np.float32)
        strike = np.array([-0.20, -0.95, 0.38], dtype=np.float32)
        follow = np.array([-0.25, -1.00, 0.55], dtype=np.float32)
        if phase < 0.24:
            u = smoothstep(phase / 0.24)
            action[3:6] = (1.0 - u) * np.zeros(3, dtype=np.float32) + u * lift
        elif phase < 0.42:
            u = smoothstep((phase - 0.24) / 0.18)
            action[3:6] = (1.0 - u) * lift + u * backswing
        elif phase < 0.66:
            u = smoothstep((phase - 0.42) / 0.24)
            action[3:6] = (1.0 - u) * backswing + u * strike
        elif phase < 0.82:
            u = smoothstep((phase - 0.66) / 0.16)
            action[3:6] = (1.0 - u) * strike + u * follow
        else:
            u = smoothstep((phase - 0.82) / 0.18)
            action[3:6] = (1.0 - u) * follow
        return np.clip(action, -1.0, 1.0)

    def recovery_action(self) -> np.ndarray:
        return np.zeros(12, dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = (1.0 - self.action_filter) * self.last_action + self.action_filter * action
        self.last_action = action.astype(np.float32)
        target_q = self.default_q + self.action_scale * action
        for _ in range(self.decimation):
            q, dq = self._q_dq()
            tau = self.kp * (target_q - q) - self.kd * dq
            self.data.ctrl[:] = np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            self._apply_balance_assist()
            self.mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[:] = 0.0
        self.step_count += 1

        toe = self._toe()
        ref = self._ref()
        toe_error = float(np.linalg.norm(toe - ref))
        base_height = float(self.data.xpos[self.base_body_id, 2])
        ball_xy = self._ball_xy()
        goal_dist = float(np.linalg.norm(ball_xy - self.goal_xy))
        goal_progress = self.prev_goal_dist - goal_dist
        self.prev_goal_dist = goal_dist
        stand_reward = np.exp(-140.0 * (base_height - 0.25) * (base_height - 0.25))
        support_action_cost = float(np.mean(np.square(action[self.support_mask])))
        fr_action_cost = float(np.mean(np.square(action[3:6])))
        healthy = base_height > 0.20
        toe_reward = np.exp(-18.0 * toe_error * toe_error)
        if not healthy:
            toe_reward *= 0.15
        expert = self.expert_action()
        imitation_reward = np.exp(-2.5 * float(np.mean(np.square(action[3:6] - expert[3:6]))))
        reward = float(
            toe_reward
            + 1.40 * stand_reward
            + 0.80 * imitation_reward
            + 12.00 * goal_progress
            + 0.05
            + 0.25 * np.exp(-1.5 * goal_dist * goal_dist)
            - 0.01 * fr_action_cost
            - 0.20 * support_action_cost
        )
        fallen = base_height < 0.19 or base_height > 0.62
        done = fallen or self.step_count >= self.horizon
        if fallen:
            reward -= 8.0
        return self._obs(), reward, done, Go2StepInfo(toe_error, base_height, ball_xy, goal_dist).__dict__

    def settle(self, steps: int = 80):
        """Let the robot settle into the default stand before an episode rollout."""
        old_filter = self.action_filter
        self.action_filter = 0.15
        for _ in range(steps):
            self.step(np.zeros(12, dtype=np.float32))
        self.step_count = 0
        self.last_action[:] = 0.0
        self.prev_goal_dist = float(np.linalg.norm(self._ball_xy() - self.goal_xy))
        self.start_toe = self._toe().copy()
        self.action_filter = old_filter
        return self._obs()

    def _apply_balance_assist(self):
        if not self.balance_assist:
            return
        z = float(self.data.xpos[self.base_body_id, 2])
        vz = float(self.data.cvel[self.base_body_id, 5])
        force_z = self.total_mass * 9.81 + 900.0 * (self.base_height_target - z) - 90.0 * vz
        self.data.xfrc_applied[self.base_body_id, 5] = float(np.clip(force_z, -120.0, 260.0))


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
