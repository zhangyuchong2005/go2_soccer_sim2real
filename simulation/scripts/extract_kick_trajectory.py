"""Extract FR leg joint trajectory from IsaacLab sim2sim kick run.

This script runs the same IsaacLab sim2sim episode as
sim2sim_isaaclab_go2.py, but records the 12 joint position targets at
every control step. The output is a JSON file containing the trajectory
in both IsaacLab order (FL/FR/RL/RR) and rl_sar SDK order (FR/FL/RR/RL).

Usage:
    cd /home/yczhang/IsaacLab-2.1.0
    TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
        -p /home/yczhang/hrl_soccer_shooting/scripts/extract_kick_trajectory.py \
        --headless \
        --episodes 1 \
        --output /home/yczhang/hrl_soccer_shooting/docs/kick_trajectory.json

The JSON output can be used to:
1. Replace hardcoded keyframes in rl_sar soccer_kick/config.yaml with
   more precise values extracted from the actual IsaacLab kick.
2. Visualize the trajectory shape for debugging.
3. Compare trajectories across different skill-speed / ball-x settings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ISAACLAB_ROOT = Path("/home/yczhang/IsaacLab-2.1.0")
if ISAACLAB_ROOT.exists():
    for source_dir in ISAACLAB_ROOT.glob("source/*"):
        if source_dir.is_dir():
            sys.path.insert(0, str(source_dir))

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser(description="Extract kick trajectory from IsaacLab sim2sim.")
parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "go2_bc_swing_visible.pt")
parser.add_argument("--controller", choices=["policy", "expert"], default="policy")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--horizon", type=int, default=220)
parser.add_argument("--skill-speed", type=float, default=4.0)
parser.add_argument("--settle-steps", type=int, default=80)
parser.add_argument("--ball-x", type=float, default=0.44)
parser.add_argument("--ball-y", type=float, default=-0.150)
parser.add_argument("--ball-z", type=float, default=0.105)
parser.add_argument("--base-z", type=float, default=0.29)
parser.add_argument("--center-hit-lift", type=float, default=0.20)
parser.add_argument("--ordered-swing", type=float, default=1.0)
parser.add_argument("--action-scale", type=float, default=1.0)
parser.add_argument("--asset-source", choices=["local-urdf", "local-mjcf", "isaaclab-usd"], default="local-urdf")
parser.add_argument("--mjcf", type=Path, default=ROOT / "assets" / "unitree_go2" / "go2.xml")
parser.add_argument("--urdf", type=Path, default=Path("/home/yczhang/unitree_rl_gym/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", type=Path, default=ROOT / "assets" / "isaaclab")
parser.add_argument("--output", type=Path, default=ROOT / "docs" / "kick_trajectory.json",
                    help="Output JSON path for the trajectory data.")
parser.add_argument("--progress-every", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch
from torch import nn

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg, UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.math import quat_apply
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


MUJOCO_GO2_JOINT_ORDER = [
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

DEFAULT_Q_MUJOCO = np.array(
    [
        0.0, 0.65, -1.45,
        0.0, 0.9, -1.8,
        0.0, 0.65, -1.45,
        0.0, 0.65, -1.45,
    ],
    dtype=np.float32,
)
ACTION_SCALE_MUJOCO = np.array([0.35, 0.55, 0.65] * 4, dtype=np.float32)
ACTION_SCALE_MUJOCO[3:6] = np.array([0.60, 1.40, 1.05], dtype=np.float32)
GOAL_XY = np.array([2.3, 0.0], dtype=np.float32)

# Mapping from MuJoCo/IsaacLab order (FL/FR/RL/RR) to rl_sar SDK order (FR/FL/RR/RL)
# MuJoCo index 0 (FL_hip) -> SDK index 3
# MuJoCo index 1 (FL_thigh) -> SDK index 4
# MuJoCo index 2 (FL_calf) -> SDK index 5
# MuJoCo index 3 (FR_hip) -> SDK index 0
# MuJoCo index 4 (FR_thigh) -> SDK index 1
# MuJoCo index 5 (FR_calf) -> SDK index 2
# MuJoCo index 6 (RL_hip) -> SDK index 9
# MuJoCo index 7 (RL_thigh) -> SDK index 10
# MuJoCo index 8 (RL_calf) -> SDK index 11
# MuJoCo index 9 (RR_hip) -> SDK index 6
# MuJoCo index 10 (RR_thigh) -> SDK index 7
# MuJoCo index 11 (RR_calf) -> SDK index 8
MUJOCO_TO_SDK_ORDER = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# rl_sar SDK joint names in order
RL_SAR_SDK_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


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

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _ = self.forward(obs_t)
        return action.squeeze(0).cpu().numpy()


def load_mujoco_policy(path: Path) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu")
    model = ActorCritic(int(ckpt["obs_dim"]), int(ckpt["act_dim"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def skill_phase(step_count: int, horizon: int) -> float:
    return min(1.0, step_count * args_cli.skill_speed / max(1, horizon - 1))


def expert_action(step_count: int, horizon: int) -> np.ndarray:
    phase = skill_phase(step_count, horizon)
    action = np.zeros(12, dtype=np.float32)
    lift = np.array([0.00, 0.35, -0.90], dtype=np.float32)
    backswing = np.array([0.00, 0.60, -1.00], dtype=np.float32)
    strike = np.array([-0.20, -0.95, 0.38], dtype=np.float32)
    follow = np.array([-0.25, -1.00, 0.55], dtype=np.float32)
    if phase < 0.24:
        u = smoothstep(phase / 0.24)
        action[3:6] = u * lift
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


def apply_ordered_swing(target_q: np.ndarray, phase: float, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return target_q
    start = DEFAULT_Q_MUJOCO[3:6]
    backswing = np.array([0.00, 1.60, -2.65], dtype=np.float32)
    strike = np.array([-0.12, -0.38, -1.22], dtype=np.float32)
    follow = np.array([-0.16, -0.55, -1.15], dtype=np.float32)
    if phase < 0.30:
        u = smoothstep(phase / 0.30)
        swing_q = (1.0 - u) * start + u * backswing
    elif phase < 0.56:
        u = smoothstep((phase - 0.30) / 0.26)
        swing_q = (1.0 - u) * backswing + u * strike
    elif phase < 0.72:
        u = smoothstep((phase - 0.56) / 0.16)
        swing_q = (1.0 - u) * strike + u * follow
    else:
        u = 1.0 - smoothstep((phase - 0.72) / 0.16)
        swing_q = u * follow + (1.0 - u) * target_q[3:6]
    out = target_q.copy()
    blend = float(np.clip(strength, 0.0, 1.0))
    out[3:6] = (1.0 - blend) * target_q[3:6] + blend * swing_q
    return out


def make_go2_cfg() -> ArticulationCfg:
    if args_cli.asset_source == "isaaclab-usd":
        go2_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/Origin/Robot")
    elif args_cli.asset_source == "local-mjcf":
        enable_extension("isaacsim.asset.importer.mjcf")
        args_cli.usd_dir.mkdir(parents=True, exist_ok=True)
        converter_cfg = MjcfConverterCfg(
            asset_path=str(args_cli.mjcf),
            usd_dir=str(args_cli.usd_dir),
            usd_file_name="go2_from_mujoco_fixed.usd",
            fix_base=True,
            import_sites=True,
            force_usd_conversion=False,
            make_instanceable=False,
        )
        converter = MjcfConverter(converter_cfg)
        usd_path = converter.usd_path
        prim_path = "/World/Origin/Robot/worldBody"
    else:
        enable_extension("isaacsim.asset.importer.urdf")
        args_cli.usd_dir.mkdir(parents=True, exist_ok=True)
        converter_cfg = UrdfConverterCfg(
            asset_path=str(args_cli.urdf),
            usd_dir=str(args_cli.usd_dir),
            usd_file_name="go2_from_urdf_feet_fixed.usd",
            fix_base=True,
            merge_fixed_joints=False,
            force_usd_conversion=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=35.0, damping=0.9),
                target_type="position",
            ),
        )
        converter = UrdfConverter(converter_cfg)
        usd_path = converter.usd_path
        prim_path = "/World/Origin/Robot"

    if args_cli.asset_source != "isaaclab-usd":
        go2_cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.45),
                joint_pos={
                    ".*_hip_joint": 0.0,
                    ".*_thigh_joint": 0.8,
                    ".*_calf_joint": -1.5,
                },
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.9,
            actuators={
                "base_legs": DCMotorCfg(
                    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                    effort_limit=23.5,
                    saturation_effort=23.5,
                    velocity_limit=30.0,
                    stiffness=35.0,
                    damping=0.9,
                    friction=0.0,
                ),
            },
        )
    go2_cfg.init_state.pos = (0.0, 0.0, args_cli.base_z)
    return go2_cfg


def design_scene() -> tuple[Articulation, RigidObject]:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))
    robot = Articulation(make_go2_cfg())
    ball_cfg = RigidObjectCfg(
        prim_path="/World/Origin/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.105,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.02,
                angular_damping=0.02,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.82, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(args_cli.ball_x, args_cli.ball_y, args_cli.ball_z)),
    )
    ball = RigidObject(ball_cfg)
    return robot, ball


def resolve_toe_body(robot: Articulation) -> tuple[int, bool]:
    for pattern in ["FR_foot", "FR.*foot.*", "FR_calf", "FR.*calf.*"]:
        try:
            ids, names = robot.find_bodies(pattern, preserve_order=True)
        except ValueError:
            continue
        if ids:
            return ids[0], "foot" in names[0].lower()
    raise RuntimeError(f"Could not find FR foot/calf body. Available: {robot.body_names}")


def build_joint_mapping(robot: Articulation) -> list[int]:
    joint_ids, joint_names = robot.find_joints(MUJOCO_GO2_JOINT_ORDER, preserve_order=True)
    return joint_ids


def toe_position(robot: Articulation, toe_body_id: int, toe_body_is_foot: bool) -> torch.Tensor:
    body_state = robot.data.body_state_w[0, toe_body_id]
    if toe_body_is_foot:
        return body_state[:3]
    local_foot = torch.tensor([[0.0, 0.0, -0.213]], device=body_state.device, dtype=body_state.dtype)
    return body_state[:3] + quat_apply(body_state[3:7].unsqueeze(0), local_foot).squeeze(0)


def write_robot_reset(robot: Articulation, joint_ids: list[int], device: str):
    root_state = robot.data.default_root_state.clone()
    root_state[:, :7] = torch.tensor([[0.0, 0.0, args_cli.base_z, 1.0, 0.0, 0.0, 0.0]], device=device)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = torch.as_tensor(DEFAULT_Q_MUJOCO, device=device).unsqueeze(0)
    joint_vel[:, :] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def write_ball_reset(ball: RigidObject, device: str):
    root_state = ball.data.default_root_state.clone()
    root_state[:, :7] = torch.tensor([[args_cli.ball_x, args_cli.ball_y, args_cli.ball_z, 1.0, 0.0, 0.0, 0.0]], device=device)
    root_state[:, 7:] = 0.0
    ball.write_root_pose_to_sim(root_state[:, :7])
    ball.write_root_velocity_to_sim(root_state[:, 7:])
    ball.reset()


def mujoco_to_sdk(q_mujoco: np.ndarray) -> np.ndarray:
    """Convert 12D joint vector from MuJoCo order (FL/FR/RL/RR) to rl_sar SDK order (FR/FL/RR/RL)."""
    q_sdk = np.zeros(12, dtype=np.float32)
    for i in range(12):
        q_sdk[MUJOCO_TO_SDK_ORDER[i]] = q_mujoco[i]
    return q_sdk


def run_extract_episode(
    sim: SimulationContext,
    robot: Articulation,
    ball: RigidObject,
    policy: ActorCritic,
    joint_ids: list[int],
    toe_body_id: int,
    toe_body_is_foot: bool,
    episode: int,
) -> list[dict]:
    """Run one kick episode and record the target joint trajectory at each control step."""
    device = sim.device
    sim_dt = sim.get_physics_dt()
    control_decimation = max(1, int(round(0.02 / sim_dt)))
    action_filter = 0.25
    last_action = np.zeros(12, dtype=np.float32)

    write_robot_reset(robot, joint_ids, device)
    write_ball_reset(ball, device)
    robot.write_data_to_sim()
    ball.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)
    ball.update(sim_dt)

    # Settle phase (no trajectory recording)
    zero_target = torch.as_tensor(DEFAULT_Q_MUJOCO, device=device).unsqueeze(0)
    for _ in range(args_cli.settle_steps * control_decimation):
        robot.set_joint_position_target(zero_target, joint_ids=joint_ids)
        robot.write_data_to_sim()
        ball.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        ball.update(sim_dt)

    start_toe = toe_position(robot, toe_body_id, toe_body_is_foot).detach().cpu().numpy().astype(np.float32)
    ball_start = ball.data.root_state_w[0, :2].detach().cpu().numpy().astype(np.float32)

    trajectory = []

    for step_count in range(args_cli.horizon):
        if args_cli.controller == "expert":
            raw_action = expert_action(step_count, args_cli.horizon)
        else:
            # Build obs for policy inference (same as sim2sim)
            q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(np.float32)
            dq = robot.data.joint_vel[0, joint_ids].detach().cpu().numpy().astype(np.float32)
            base_pos = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(np.float32)
            toe_pos = toe_position(robot, toe_body_id, toe_body_is_foot).detach().cpu().numpy().astype(np.float32)
            ball_xy = ball.data.root_state_w[0, :2].detach().cpu().numpy().astype(np.float32)
            strike_phase = skill_phase(step_count, args_cli.horizon)
            obs = np.concatenate([q, dq, base_pos, toe_pos, ball_xy, np.array([strike_phase], dtype=np.float32)])
            raw_action = policy.act(obs)

        raw_action = np.clip(raw_action * args_cli.action_scale, -1.0, 1.0)
        action = (1.0 - action_filter) * last_action + action_filter * raw_action
        last_action = action.astype(np.float32)
        target_q_mujoco = DEFAULT_Q_MUJOCO + ACTION_SCALE_MUJOCO * action

        strike_phase = skill_phase(step_count, args_cli.horizon)
        target_q_mujoco = apply_ordered_swing(target_q_mujoco, strike_phase, args_cli.ordered_swing)

        if args_cli.center_hit_lift != 0.0 and strike_phase <= 0.58:
            lift_in = smoothstep(strike_phase / 0.08)
            lift_out = 1.0 - smoothstep((strike_phase - 0.46) / 0.12)
            lift_weight = max(0.0, min(lift_in, lift_out))
            target_q_mujoco[5] -= args_cli.center_hit_lift * lift_weight

        # Convert to rl_sar SDK order and record
        target_q_sdk = mujoco_to_sdk(target_q_mujoco)

        trajectory.append({
            "step": step_count,
            "phase": float(strike_phase),
            "time_s": float(step_count * 0.02),
            "target_q_mujoco": target_q_mujoco.tolist(),
            "target_q_sdk": target_q_sdk.tolist(),
            "action_mujoco": action.tolist(),
        })

        target_q_t = torch.as_tensor(target_q_mujoco, dtype=torch.float32, device=device).unsqueeze(0)
        for _ in range(control_decimation):
            robot.set_joint_position_target(target_q_t, joint_ids=joint_ids)
            robot.write_data_to_sim()
            ball.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
            ball.update(sim_dt)

        if not simulation_app.is_running():
            return trajectory

    # Final metrics
    ball_end = ball.data.root_state_w[0, :2].detach().cpu().numpy().astype(np.float32)
    ball_move = float(np.linalg.norm(ball_end - ball_start))
    goal_dist = float(np.linalg.norm(ball_end - GOAL_XY))
    base_h = float(robot.data.root_state_w[0, 2].item())
    print(f"episode={episode:02d} ball_move={ball_move:.3f} goal_dist={goal_dist:.3f} min_base_h={base_h:.3f}")

    return trajectory


def extract_keyframes_from_trajectory(trajectory: list[dict]) -> dict:
    """Extract keyframe angles from trajectory at the phase boundaries used by rl_sar SoccerKick.

    rl_sar SoccerKick phase boundaries (from config.yaml): [0.35, 0.65, 0.80]
    These correspond to: backswing peak, strike, follow-through.
    """
    # Find the closest trajectory samples at key phases
    target_phases = {
        "backswing": 0.30,  # Peak of backswing (before strike starts at 0.35)
        "strike": 0.50,     # Mid-strike (between 0.35 and 0.65)
        "follow": 0.72,     # Mid-follow (between 0.65 and 0.80)
    }

    keyframes = {}
    for name, target_phase in target_phases.items():
        best = None
        best_dist = float("inf")
        for t in trajectory:
            dist = abs(t["phase"] - target_phase)
            if dist < best_dist:
                best_dist = dist
                best = t
        if best is not None:
            keyframes[name] = {
                "phase": best["phase"],
                "time_s": best["time_s"],
                "fr_sdk": best["target_q_sdk"][:3],  # FR leg in SDK order
                "all_sdk": best["target_q_sdk"],
            }

    return keyframes


def main():
    sys.path.insert(0, str(ROOT))
    policy = load_mujoco_policy(args_cli.checkpoint)

    sim_cfg = sim_utils.SimulationCfg(dt=0.002, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, -1.4, 0.8], target=[0.35, -0.08, 0.18])

    robot, ball = design_scene()
    sim.reset()
    print("[INFO] IsaacLab setup complete")

    joint_ids = build_joint_mapping(robot)
    toe_body_id, toe_body_is_foot = resolve_toe_body(robot)

    all_trajectories = []
    for episode in range(1, args_cli.episodes + 1):
        if not simulation_app.is_running():
            break
        traj = run_extract_episode(sim, robot, ball, policy, joint_ids, toe_body_id, toe_body_is_foot, episode)
        all_trajectories.append(traj)
        time.sleep(0.1)

    # Use the last episode for keyframe extraction
    if all_trajectories:
        last_traj = all_trajectories[-1]
        keyframes = extract_keyframes_from_trajectory(last_traj)

        output_data = {
            "metadata": {
                "checkpoint": str(args_cli.checkpoint),
                "skill_speed": args_cli.skill_speed,
                "ball_x": args_cli.ball_x,
                "center_hit_lift": args_cli.center_hit_lift,
                "ordered_swing": args_cli.ordered_swing,
                "horizon": args_cli.horizon,
                "num_episodes": args_cli.episodes,
            },
            "keyframes_sdk_order": keyframes,
            "joint_order_sdk": RL_SAR_SDK_JOINT_NAMES,
            "trajectory_last_episode": last_traj,
        }

        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args_cli.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"[INFO] Trajectory saved to {args_cli.output}")
        print(f"[INFO] Extracted keyframes:")
        for name, kf in keyframes.items():
            fr = kf["fr_sdk"]
            print(f"  {name}: phase={kf['phase']:.3f} FR=[hip={fr[0]:.3f}, thigh={fr[1]:.3f}, calf={fr[2]:.3f}]")

        # Print suggested config.yaml update
        print("\n[SUGGESTED] Update soccer_kick/config.yaml with extracted keyframes:")
        for name in ["backswing", "strike", "follow"]:
            if name in keyframes:
                fr = keyframes[name]["fr_sdk"]
                print(f"  fr_{name}: [{fr[0]:.2f}, {fr[1]:.2f}, {fr[2]:.2f}]")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if args_cli.headless:
            os._exit(0)
        simulation_app.close()