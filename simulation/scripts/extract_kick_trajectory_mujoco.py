"""Extract kick trajectory from MuJoCo sim using the trained policy.

Records 12 joint position targets at every control step during a kick
episode, then extracts keyframes at phase boundaries for rl_sar deployment.

Usage:
    cd /home/yczhang/hrl_soccer_shooting
    conda run --no-capture-output -n mujoco-go2 python scripts/extract_kick_trajectory_mujoco.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from hrl_soccer.mujoco_go2 import GO2_SCENE, Go2BezierKickEnv
from hrl_soccer.ppo import load_policy

CHECKPOINT = ROOT / "checkpoints" / "go2_bc_swing_visible.pt"
OUTPUT = ROOT / "docs" / "kick_trajectory.json"

# rl_sar SDK joint order: FR/FL/RR/RL
MUJOCO_TO_SDK = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
SDK_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def mujoco_to_sdk(q: np.ndarray) -> np.ndarray:
    out = np.zeros(12, dtype=np.float32)
    for i in range(12):
        out[MUJOCO_TO_SDK[i]] = q[i]
    return out


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def apply_ordered_swing(target_q: np.ndarray, phase: float) -> np.ndarray:
    """Same as IsaacLab script — apply FR leg ordered swing guard.
    Keyframes match expert_action() in mujoco_go2.py.
    """
    DEFAULT_Q = np.array([0.0, 0.65, -1.45, 0.0, 0.9, -1.8, 0.0, 0.65, -1.45, 0.0, 0.65, -1.45], dtype=np.float32)
    start = DEFAULT_Q[3:6]
    backswing = np.array([0.00, 1.00, -1.80], dtype=np.float32)
    strike = np.array([-0.20, -0.95, 0.38], dtype=np.float32)
    follow = np.array([-0.25, -1.00, 0.55], dtype=np.float32)
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
    out[3:6] = swing_q
    return out


def main():
    policy = load_policy(CHECKPOINT)
    env = Go2BezierKickEnv(GO2_SCENE, seed=303)

    HORIZON = 180
    SETTLE = 80

    obs = env.reset()
    obs = env.settle(SETTLE)
    ball_start = env._ball_xy().copy()

    trajectory = []
    for step in range(HORIZON):
        action, _, _ = policy.act(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        # Reconstruct what the env does internally to get target_q
        ACTION_SCALE = np.array([0.35, 0.55, 0.65] * 4, dtype=np.float32)
        ACTION_SCALE[3:6] = np.array([0.60, 1.40, 1.05], dtype=np.float32)
        DEFAULT_Q = np.array([0.0, 0.65, -1.45, 0.0, 0.9, -1.8, 0.0, 0.65, -1.45, 0.0, 0.65, -1.45], dtype=np.float32)

        target_q = DEFAULT_Q + ACTION_SCALE * action
        phase = min(1.0, step * 4.0 / max(1, HORIZON - 1))
        target_q = apply_ordered_swing(target_q, phase)

        # Apply center-hit-lift (calf retraction during close-ball contact)
        if phase <= 0.58:
            lift_in = smoothstep(phase / 0.08)
            lift_out = 1.0 - smoothstep((phase - 0.46) / 0.12)
            lift_weight = max(0.0, min(lift_in, lift_out))
            target_q[5] -= 0.20 * lift_weight

        target_sdk = mujoco_to_sdk(target_q)
        trajectory.append({
            "step": step,
            "phase": float(phase),
            "time_s": float(step * 0.02),
            "target_q_mujoco": target_q.tolist(),
            "target_q_sdk": target_sdk.tolist(),
        })

        obs, reward, done, info = env.step(action)
        # Pin base like play_go2_policy.py
        env.data.qpos[:7] = [0.0, 0.0, 0.29, 1.0, 0.0, 0.0, 0.0]
        env.data.qvel[:6] = 0.0
        env.mujoco.mj_forward(env.model, env.data)
        obs = env._obs()
        # Note: policy was trained with pinned base. Float base observations
        # differ from training distribution — the policy may behave poorly.
        # Use --pin-base on play_go2_policy.py for consistent results.

        if (step + 1) % 30 == 0:
            print(f"  step={step + 1}/{HORIZON}")

    ball_end = info["ball_xy"]
    ball_move = float(np.linalg.norm(ball_end - ball_start))
    goal_dist = float(np.linalg.norm(ball_end - np.array([2.3, 0.0], dtype=np.float32)))
    print(f"ball_move={ball_move:.3f} goal_dist={goal_dist:.3f}")

    # Extract keyframes
    keyframes = {}
    for name, target_phase in [("backswing", 0.30), ("strike", 0.50), ("follow", 0.72)]:
        best = min(trajectory, key=lambda t: abs(t["phase"] - target_phase))
        fr = best["target_q_sdk"][:3]
        keyframes[name] = {
            "phase": best["phase"],
            "time_s": best["time_s"],
            "fr_sdk": fr,
            "all_sdk": best["target_q_sdk"],
        }

    output_data = {
        "metadata": {
            "checkpoint": str(CHECKPOINT),
            "source": "mujoco",
            "horizon": HORIZON,
            "skill_speed": 4.0,
        },
        "keyframes_sdk_order": keyframes,
        "joint_order_sdk": SDK_NAMES,
        "trajectory_last_episode": trajectory,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[INFO] Trajectory saved to {OUTPUT}")
    print(f"[INFO] Extracted keyframes:")
    for name, kf in keyframes.items():
        fr = kf["fr_sdk"]
        print(f"  {name}: phase={kf['phase']:.3f} FR=[hip={fr[0]:.3f}, thigh={fr[1]:.3f}, calf={fr[2]:.3f}]")
    print("\n[SUGGESTED] Update soccer_kick/config.yaml with extracted keyframes:")
    for name in ["backswing", "strike", "follow"]:
        if name in keyframes:
            fr = keyframes[name]["fr_sdk"]
            print(f"  fr_{name}: [{fr[0]:.2f}, {fr[1]:.2f}, {fr[2]:.2f}]")


if __name__ == "__main__":
    main()
