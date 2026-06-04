# Go2 Soccer Kick Sim2Real with rl_sar

This document describes how to deploy the IsaacLab-verified Go2 soccer kick
onto the real robot using the rl_sar open-loop FSM state.

## Pipeline Overview

```text
IsaacLab sim2sim (verified kick)
    → extract_kick_trajectory.py (record joint angles)
    → soccer_kick/config.yaml (YAML keyframe parameters)
    → RLFSMStateSoccerKick (rl_sar FSM state, reads YAML)
    → rl_real_go2 (C++ binary, runs on real Go2)
```

The deployed kick is **open-loop keyframe animation**, not a neural-network
policy. This is intentional: the MuJoCo/IsaacLab policy uses observations
(toe position, ball position, reference trajectory) that are unavailable on
the real robot without external perception. The keyframe approach requires no
sensors beyond the standard Go2 IMU and motor encoders.

## Quick Start

### 1. Build

```bash
cd /home/yczhang/rl_sar
cmake src/rl_sar/ -B cmake_build -DUSE_CMAKE=ON
cmake --build cmake_build --target rl_real_go2 -j4
```

### 2. Install Config

```bash
/home/yczhang/hrl_soccer_shooting/scripts/install_soccer_kick_to_rl_sar.sh
```

### 3. Run on Real Go2

```bash
# Find the network interface connected to the Go2
ip -br addr

# Start the controller
cd /home/yczhang/rl_sar
./cmake_build/bin/rl_real_go2 enp4s0
```

### 4. Trigger the Kick

| Key    | Action                    |
|--------|---------------------------|
| `0`    | Stand up / GetUp          |
| `2`    | SoccerKick (after GetUp)  |
| `P`    | Emergency Passive         |
| `1`    | RL Locomotion (walking)   |
| `9`    | GetDown (sit down)        |

Gamepad: RB + DPadDown = SoccerKick, LB + X = Passive.

The kick state **automatically returns to GetUp** after `auto_return_time`
seconds (default 1.15s).

## Safety-First Testing Sequence

### First Test (No Ball)

1. Suspend the robot or place it on flat ground with space in front of FR leg.
2. Set `safety_scale: 0.5` in `soccer_kick/config.yaml`.
3. Start `rl_real_go2`.
4. Press `0` to enter GetUp. Wait until the robot finishes standing.
5. Press `2` to trigger SoccerKick.
6. **Press `P` immediately** if the robot leans, slips, or behaves unexpectedly.

Check:
- FR leg swings backward before swinging forward (backswing visible)
- The body does not roll sharply to the right
- The other three legs stay planted
- `P` immediately puts the robot into passive mode

### Gradual Increase

After successful tests at `safety_scale: 0.5`:

1. Increase to `safety_scale: 0.75` — rebuild not needed, only edit YAML.
2. Increase to `safety_scale: 1.0` — full-amplitude kick.
3. Only then test with a light ball close to the FR foot.

**Never set `safety_scale > 1.0` unless you have verified the full kick
in simulation first.**

## Config Parameters

All kick parameters are in:

```text
/home/yczhang/rl_sar/policy/go2/soccer_kick/config.yaml
```

Editing this file does **not** require rebuilding — the FSM reads it on
every state entry.

### Key Parameters

| Parameter            | Default   | Description                                        |
|----------------------|-----------|----------------------------------------------------|
| `kick_duration`      | 1.00      | Total kick duration (seconds)                      |
| `auto_return_time`   | 1.15      | Time before auto-return to GetUp (seconds)         |
| `safety_scale`       | 1.0       | Amplitude scale (0.5 = half-power for first test)  |
| `kick_kp`            | 45.0      | PD position stiffness (all 12 joints)              |
| `kick_kd`            | 5.0       | PD velocity damping (all 12 joints)                |
| `fr_hip_locked`      | true      | Lock FR hip to prevent lateral sweep               |
| `roll_threshold`     | 30.0      | Attitude protect: max roll (degrees)               |
| `pitch_threshold`    | 30.0      | Attitude protect: max pitch (degrees)              |

### FR Leg Keyframes (SDK order: FR_hip, FR_thigh, FR_calf)

| Keyframe    | Default (hip, thigh, calf)  | Description           |
|-------------|----------------------------|-----------------------|
| `fr_backswing` | [0.0, 1.35, -2.35]       | Peak backswing pose   |
| `fr_strike`    | [0.0, -0.28, -2.05]      | Strike through ball   |
| `fr_follow`    | [0.0, -0.34, -1.95]      | Follow-through pose   |

- `hip=0.0` means "lock to measured start angle" — the FSM replaces it at
  runtime with the actual standing hip angle. Set `fr_hip_locked: false`
  to allow hip swing (side kicks).

### Phase Boundaries

```yaml
phase_boundaries: [0.35, 0.65, 0.80]
```

- `0.00 → 0.35`: backswing (FR leg swings backward)
- `0.35 → 0.65`: strike (FR leg sweeps forward through ball)
- `0.65 → 0.80`: follow-through
- `0.80 → 1.00`: return to standing pose

### safety_scale Mechanics

`safety_scale` scales the deviation from the standing pose:

```text
target = standing + safety_scale * (keyframe - standing)
```

- `safety_scale = 0.5`: the FR leg only travels halfway between standing
  and the full keyframe. The trajectory shape is preserved, but amplitude
  is reduced.
- `safety_scale = 1.0`: full-amplitude kick.
- `safety_scale = 0.0`: FR leg stays at standing pose (no kick at all).

## Extracting More Precise Keyframes from IsaacLab

The default keyframes are manually chosen conservative values. To extract
more precise keyframes from the actual IsaacLab simulation:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
    -p /home/yczhang/hrl_soccer_shooting/scripts/extract_kick_trajectory.py \
    --headless \
    --episodes 1 \
    --output /home/yczhang/hrl_soccer_shooting/docs/kick_trajectory.json
```

The script records every control step's target joint angles and extracts
keyframes at the phase boundaries. The output JSON includes:

- `keyframes_sdk_order`: suggested keyframes in rl_sar SDK order
- `trajectory_last_episode`: full time-series of joint targets
- `metadata`: IsaacLab settings used

You can use the extracted keyframes to replace the defaults in
`soccer_kick/config.yaml`.

## Tuning Guide

### Kick Too Weak / Ball Doesn't Move

1. Increase `safety_scale` (up to 1.0).
2. Decrease `kick_kd` (e.g., 3.0) — less damping means faster swing.
3. Increase `kick_duration` to 0.0 (not recommended) or adjust `fr_strike`
   thigh to be more negative (e.g., -0.35).

### Robot Leans / Rolls Too Much

1. Increase `kick_kd` (e.g., 8.0) — more damping slows the swing.
2. Increase `kick_duration` (e.g., 1.20) — slower kick is more stable.
3. Reduce `fr_strike` thigh magnitude (e.g., -0.18 instead of -0.28).
4. Check that `fr_hip_locked` is `true`.

### Foot Hits the Ball Too Low (Ground Rub)

1. Increase `fr_strike` calf (more negative means more retracted, e.g., -2.20).
2. Increase `fr_follow` calf similarly (e.g., -2.10).

### Foot Hits Too High / Misses the Ball

1. Decrease `fr_strike` calf (closer to standing, e.g., -1.80).
2. Decrease `fr_follow` calf.

### Attitude Protection Triggers (Roll > 30°)

1. The kick is too aggressive. Reduce `safety_scale` first.
2. If you need to increase the threshold for a specific test, edit
   `roll_threshold` / `pitch_threshold` in config.yaml. **Do not exceed
   45 degrees** — that risks the robot falling over.

## AttitudeProtect

The SoccerKick FSM calls `rl.AttitudeProtect()` on every control step.
If the robot's roll or pitch exceeds the configured thresholds, the FSM
immediately transitions to `RLFSMStatePassive` (damping mode, all motors
disengage). This is a hard safety stop — the kick is aborted mid-swing.

This protection is **always active** regardless of `safety_scale`.

## Comparison with Walking Policy Deployment

| Aspect              | Walking (RL Locomotion)     | Soccer Kick (Open-loop)       |
|----------------------|----------------------------|-------------------------------|
| Policy type          | Neural network (policy.pt) | Keyframe animation            |
| Observations         | 45D (IMU + joints + cmd)   | None (uses only standing pose)|
| Observation history  | Optional                   | None                          |
| Joint mapping        | Yes (training → SDK order) | No (already in SDK order)     |
| PD gains             | rl_kp/rl_kd from YAML      | kick_kp/kick_kd from YAML     |
| Config path          | go2/isaaclab_no_history/    | go2/soccer_kick/              |
| Trigger              | Key `1`                    | Key `2`                       |
| Safety               | AttitudeProtect optional   | AttitudeProtect always on     |

## Do Not

- Run this while the Go2 is in sport mode.
- Start with the robot near walls, people, or fragile objects.
- Set `safety_scale > 1.0` without sim verification first.
- Disable `fr_hip_locked` unless you intentionally want a side kick.
- Use the raw MuJoCo/IsaacLab neural-network policy directly on hardware.
- Exceed `roll_threshold` / `pitch_threshold` of 45 degrees.