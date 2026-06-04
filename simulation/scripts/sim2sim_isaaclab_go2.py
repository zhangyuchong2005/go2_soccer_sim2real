from __future__ import annotations

import argparse
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


parser = argparse.ArgumentParser(description="Run a MuJoCo-trained Go2 soccer policy in IsaacLab.")
parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "go2_bc_swing_visible.pt")
parser.add_argument("--controller", choices=["policy", "expert"], default="policy")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--horizon", type=int, default=180)
parser.add_argument(
    "--skill-speed",
    type=float,
    default=4.0,
    help="Speed multiplier for the kicking skill phase. 4.0 makes the motion about four times faster.",
)
parser.add_argument("--settle-steps", type=int, default=80)
parser.add_argument("--progress-every", type=int, default=30)
parser.add_argument("--action-scale", type=float, default=1.0, help="Extra safety multiplier on normalized policy actions.")
parser.add_argument(
    "--ball-x",
    type=float,
    default=0.44,
    help="Ball x position. Larger values delay contact so the foot hits closer to the ball center height.",
)
parser.add_argument("--ball-y", type=float, default=-0.150)
parser.add_argument("--ball-z", type=float, default=0.105)
parser.add_argument("--base-z", type=float, default=0.29)
parser.add_argument(
    "--kick-height-bias",
    type=float,
    default=0.0,
    help="Legacy FR calf joint target bias in radians during the strike.",
)
parser.add_argument(
    "--center-hit-lift",
    type=float,
    default=0.20,
    help="Early FR calf retraction in radians. Raises the foot before close-ball contact.",
)
parser.add_argument(
    "--ordered-swing",
    type=float,
    default=1.0,
    help="Blend in an explicit backswing-before-strike FR leg timing guard. Use 0 to disable.",
)
parser.add_argument("--asset-source", choices=["local-urdf", "local-mjcf", "isaaclab-usd"], default="local-urdf")
parser.add_argument("--mjcf", type=Path, default=ROOT / "assets" / "unitree_go2" / "go2.xml")
parser.add_argument("--urdf", type=Path, default=Path("/home/yczhang/unitree_rl_gym/resources/robots/go2/urdf/go2.urdf"))
parser.add_argument("--usd-dir", type=Path, default=ROOT / "assets" / "isaaclab")
parser.add_argument("--floating-base", action="store_true", help="Import the MJCF with a free base. Default is fixed base.")
parser.add_argument("--no-real-time", action="store_true", help="Disable real-time sleeps in GUI mode.")
parser.add_argument("--playback-rate", type=float, default=1.0, help="GUI playback speed multiplier.")
parser.add_argument("--hold-after", type=float, default=4.0, help="Seconds to hold the final pose in GUI mode after each episode.")
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
        0.0,
        0.65,
        -1.45,
        0.0,
        0.9,
        -1.8,
        0.0,
        0.65,
        -1.45,
        0.0,
        0.65,
        -1.45,
    ],
    dtype=np.float32,
)
ACTION_SCALE_MUJOCO = np.array([0.35, 0.55, 0.65] * 4, dtype=np.float32)
ACTION_SCALE_MUJOCO[3:6] = np.array([0.60, 1.40, 1.05], dtype=np.float32)
GOAL_XY = np.array([2.3, 0.0], dtype=np.float32)


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
        if not check_file_path(str(args_cli.mjcf)):
            raise FileNotFoundError(f"Invalid MJCF path: {args_cli.mjcf}")
        enable_extension("isaacsim.asset.importer.mjcf")
        args_cli.usd_dir.mkdir(parents=True, exist_ok=True)
        converter_cfg = MjcfConverterCfg(
            asset_path=str(args_cli.mjcf),
            usd_dir=str(args_cli.usd_dir),
            usd_file_name="go2_from_mujoco_floating.usd" if args_cli.floating_base else "go2_from_mujoco_fixed.usd",
            fix_base=not args_cli.floating_base,
            import_sites=True,
            force_usd_conversion=False,
            make_instanceable=False,
        )
        converter = MjcfConverter(converter_cfg)
        usd_path = converter.usd_path
        print(f"[INFO] using converted local MJCF USD: {usd_path}")
        prim_path = "/World/Origin/Robot/worldBody"
    else:
        if not check_file_path(str(args_cli.urdf)):
            raise FileNotFoundError(f"Invalid URDF path: {args_cli.urdf}")
        enable_extension("isaacsim.asset.importer.urdf")
        args_cli.usd_dir.mkdir(parents=True, exist_ok=True)
        converter_cfg = UrdfConverterCfg(
            asset_path=str(args_cli.urdf),
            usd_dir=str(args_cli.usd_dir),
            usd_file_name="go2_from_urdf_feet_floating.usd" if args_cli.floating_base else "go2_from_urdf_feet_fixed.usd",
            fix_base=not args_cli.floating_base,
            merge_fixed_joints=False,
            force_usd_conversion=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=35.0, damping=0.9),
                target_type="position",
            ),
        )
        converter = UrdfConverter(converter_cfg)
        usd_path = converter.usd_path
        print(f"[INFO] using converted local URDF USD: {usd_path}")
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
            print(f"[INFO] toe body: {names[0]} id={ids[0]}")
            return ids[0], "foot" in names[0].lower()
    raise RuntimeError(f"Could not find a front-right foot/calf body. Available bodies: {robot.body_names}")


def build_joint_mapping(robot: Articulation) -> list[int]:
    joint_ids, joint_names = robot.find_joints(MUJOCO_GO2_JOINT_ORDER, preserve_order=True)
    if joint_names != MUJOCO_GO2_JOINT_ORDER:
        raise RuntimeError(
            "IsaacLab Go2 joint names do not match the MuJoCo order.\n"
            f"wanted={MUJOCO_GO2_JOINT_ORDER}\nresolved={joint_names}\navailable={robot.joint_names}"
        )
    print("[INFO] MuJoCo->IsaacLab joint ids:", dict(zip(joint_names, joint_ids)))
    return joint_ids


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


def ref_trajectory(step_count: int, horizon: int, start_toe: np.ndarray, ball_xy: np.ndarray) -> np.ndarray:
    t = skill_phase(step_count, horizon)
    aim = GOAL_XY - ball_xy
    aim_dir = aim / (np.linalg.norm(aim) + 1e-6)
    back = np.array([-aim_dir[0], -aim_dir[1], 0.0], dtype=np.float32)
    impact = np.array([ball_xy[0], ball_xy[1], args_cli.ball_z], dtype=np.float32) + 0.12 * back
    follow = np.array([ball_xy[0], ball_xy[1], args_cli.ball_z + 0.01], dtype=np.float32) + 0.18 * np.r_[aim_dir, 0.0].astype(np.float32)
    lift = start_toe + np.array([0.06, -0.02, 0.16], dtype=np.float32)
    prep = impact + np.array([-0.16 * aim_dir[0], -0.16 * aim_dir[1], 0.10], dtype=np.float32)
    if t < 0.25:
        u = t / 0.25
        return (1.0 - u) * start_toe + u * lift
    if t < 0.60:
        u = (t - 0.25) / 0.35
        return (1.0 - u) * lift + u * prep
    if t < 0.82:
        u = (t - 0.60) / 0.22
        return (1.0 - u) * prep + u * impact
    u = (t - 0.82) / 0.18
    return (1.0 - u) * impact + u * follow


def make_obs(
    robot: Articulation,
    ball: RigidObject,
    joint_ids: list[int],
    toe_body_id: int,
    toe_body_is_foot: bool,
    step_count: int,
    horizon: int,
    start_toe: np.ndarray,
) -> np.ndarray:
    q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(np.float32)
    dq = robot.data.joint_vel[0, joint_ids].detach().cpu().numpy().astype(np.float32)
    base_pos = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(np.float32)
    toe_pos = toe_position(robot, toe_body_id, toe_body_is_foot).detach().cpu().numpy().astype(np.float32)
    ball_xy = ball.data.root_state_w[0, :2].detach().cpu().numpy().astype(np.float32)
    ref = ref_trajectory(step_count, horizon, start_toe, ball_xy)
    phase = np.array([skill_phase(step_count, horizon)], dtype=np.float32)
    return np.concatenate([q, dq, base_pos, toe_pos, ref, ball_xy, phase]).astype(np.float32)


def toe_position(robot: Articulation, toe_body_id: int, toe_body_is_foot: bool) -> torch.Tensor:
    body_state = robot.data.body_state_w[0, toe_body_id]
    if toe_body_is_foot:
        return body_state[:3]
    local_foot = torch.tensor([[0.0, 0.0, -0.213]], device=body_state.device, dtype=body_state.dtype)
    return body_state[:3] + quat_apply(body_state[3:7].unsqueeze(0), local_foot).squeeze(0)


def run_episode(
    sim: SimulationContext,
    robot: Articulation,
    ball: RigidObject,
    policy: ActorCritic,
    joint_ids: list[int],
    toe_body_id: int,
    toe_body_is_foot: bool,
    episode: int,
):
    device = sim.device
    sim_dt = sim.get_physics_dt()
    control_decimation = max(1, int(round(0.02 / sim_dt)))
    real_time = (not args_cli.headless) and (not args_cli.no_real_time)
    playback_rate = max(1.0e-6, args_cli.playback_rate)
    action_filter = 0.25
    last_action = np.zeros(12, dtype=np.float32)

    write_robot_reset(robot, joint_ids, device)
    write_ball_reset(ball, device)
    robot.write_data_to_sim()
    ball.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)
    ball.update(sim_dt)

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
    base_h_min = 10.0
    closest_toe_ball_xy = float("inf")
    closest_toe_z = 0.0
    impact_toe_z = None
    impact_phase = None

    for step_count in range(args_cli.horizon):
        if args_cli.progress_every > 0 and step_count % args_cli.progress_every == 0:
            print(f"[INFO] episode={episode:02d} step={step_count}/{args_cli.horizon}", flush=True)
        if args_cli.controller == "expert":
            raw_action = expert_action(step_count, args_cli.horizon)
        else:
            obs = make_obs(robot, ball, joint_ids, toe_body_id, toe_body_is_foot, step_count, args_cli.horizon, start_toe)
            raw_action = policy.act(obs)
        raw_action = np.clip(raw_action * args_cli.action_scale, -1.0, 1.0)
        action = (1.0 - action_filter) * last_action + action_filter * raw_action
        last_action = action.astype(np.float32)
        target_q = DEFAULT_Q_MUJOCO + ACTION_SCALE_MUJOCO * action
        strike_phase = skill_phase(step_count, args_cli.horizon)
        target_q = apply_ordered_swing(target_q, strike_phase, args_cli.ordered_swing)
        if args_cli.center_hit_lift != 0.0 and strike_phase <= 0.58:
            lift_in = smoothstep(strike_phase / 0.08)
            lift_out = 1.0 - smoothstep((strike_phase - 0.46) / 0.12)
            lift_weight = max(0.0, min(lift_in, lift_out))
            target_q[5] -= args_cli.center_hit_lift * lift_weight
        if args_cli.kick_height_bias != 0.0 and 0.36 <= strike_phase <= 0.82:
            target_q[5] += args_cli.kick_height_bias * smoothstep((strike_phase - 0.36) / 0.12)
        target_q_t = torch.as_tensor(target_q, dtype=torch.float32, device=device).unsqueeze(0)

        for _ in range(control_decimation):
            step_start = time.time()
            robot.set_joint_position_target(target_q_t, joint_ids=joint_ids)
            robot.write_data_to_sim()
            ball.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
            ball.update(sim_dt)
            if real_time:
                sleep_time = sim_dt / playback_rate - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        base_h = float(robot.data.root_state_w[0, 2].item())
        base_h_min = min(base_h_min, base_h)
        toe_xyz = toe_position(robot, toe_body_id, toe_body_is_foot).detach().cpu().numpy()
        ball_xyz = ball.data.root_state_w[0, :3].detach().cpu().numpy()
        toe_ball_xy = float(np.linalg.norm(toe_xyz[:2] - ball_xyz[:2]))
        if toe_ball_xy < closest_toe_ball_xy:
            closest_toe_ball_xy = toe_ball_xy
            closest_toe_z = float(toe_xyz[2])
        ball_move_now = float(np.linalg.norm(ball_xyz[:2] - ball_start))
        if impact_toe_z is None and ball_move_now > 0.02:
            impact_toe_z = float(toe_xyz[2])
            impact_phase = strike_phase
        if not simulation_app.is_running():
            return

    ball_end = ball.data.root_state_w[0, :2].detach().cpu().numpy().astype(np.float32)
    ball_move = float(np.linalg.norm(ball_end - ball_start))
    goal_dist = float(np.linalg.norm(ball_end - GOAL_XY))
    print(
        f"episode={episode:02d} ball_move={ball_move:.3f} "
        f"goal_dist={goal_dist:.3f} min_base_h={base_h_min:.3f} "
        f"closest_toe_z={closest_toe_z:.3f} "
        f"impact_toe_z={impact_toe_z if impact_toe_z is not None else float('nan'):.3f} "
        f"impact_phase={impact_phase if impact_phase is not None else float('nan'):.2f} "
        f"ball_z={args_cli.ball_z:.3f}",
        flush=True,
    )
    if real_time and args_cli.hold_after > 0.0:
        end_time = time.time() + args_cli.hold_after
        while simulation_app.is_running() and time.time() < end_time:
            step_start = time.time()
            robot.set_joint_position_target(target_q_t, joint_ids=joint_ids)
            robot.write_data_to_sim()
            ball.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
            ball.update(sim_dt)
            sleep_time = sim_dt / playback_rate - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def main():
    sys.path.insert(0, str(ROOT))
    policy = load_mujoco_policy(args_cli.checkpoint)

    sim_cfg = sim_utils.SimulationCfg(dt=0.002, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, -1.4, 0.8], target=[0.35, -0.08, 0.18])

    robot, ball = design_scene()
    sim.reset()
    print("[INFO] IsaacLab setup complete")
    print(f"[INFO] robot joints: {robot.joint_names}")
    print(f"[INFO] robot bodies: {robot.body_names}")

    joint_ids = build_joint_mapping(robot)
    toe_body_id, toe_body_is_foot = resolve_toe_body(robot)

    for episode in range(1, args_cli.episodes + 1):
        if not simulation_app.is_running():
            break
        run_episode(sim, robot, ball, policy, joint_ids, toe_body_id, toe_body_is_foot, episode)
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if args_cli.headless:
            # Isaac Sim can hang during teardown in headless conda runs after a
            # completed metric pass. Exit after flushing so automation returns.
            os._exit(0)
        simulation_app.close()
