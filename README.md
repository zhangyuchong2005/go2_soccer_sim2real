# Go2 Soccer Sim2Real

Hierarchical reinforcement learning soccer shooting skill for the Unitree Go2 quadruped robot, with simulation-to-real deployment pipeline.

> **Paper**: Yandong Ji et al. "Hierarchical Reinforcement Learning for Precise Soccer Shooting Skills using a Quadrupedal Robot", IROS 2022 / arXiv:2208.01160.
>
> This project is a compact reproduction scaffold, not the official implementation.

## Project Structure

```
go2_soccer_sim2real/
├── simulation/              # MuJoCo + IsaacLab simulation
│   ├── hrl_soccer/          # Core Python modules (envs, PPO, bezier)
│   ├── scripts/             # Training, evaluation, visualization scripts
│   ├── checkpoints/         # Trained policy weights
│   ├── assets/              # MuJoCo scenes, USD models
│   ├── configs/             # Training configs
│   └── docs/                # Trajectory JSON, documentation
└── deploy/                  # Sim2Real deployment on real Go2
    ├── rl_sar/              # Modified rl_sar FSM state (fsm_go2.hpp)
    └── policy/              # Soccer kick open-loop config (config.yaml)
```

## Quick Start — Simulation

### Prerequisites

```bash
conda create -n mujoco-go2 python=3.10
conda activate mujoco-go2
pip install numpy torch mujoco
```

### 1. Check Environment

```bash
cd simulation
conda run --no-capture-output -n mujoco-go2 python scripts/check_env.py
```

### 2. Train Policies

**Low-level controller** (tracks kicking toe trajectories):
```bash
conda run --no-capture-output -n mujoco-go2 python scripts/train_control.py --total-steps 200000
```

**High-level planner** (outputs 3x5 Bezier parameters):
```bash
conda run --no-capture-output -n mujoco-go2 python scripts/train_planner.py --total-steps 300000
```

**Behavioral cloning** (learn expert keyframe animation):
```bash
conda run --no-capture-output -n mujoco-go2 python scripts/train_go2_bc.py --episodes 300 --epochs 80 --checkpoint checkpoints/go2_bc_policy.pt
```

### 3. Evaluate in MuJoCo

```bash
conda run --no-capture-output -n mujoco-go2 python scripts/play_go2_policy.py \
  --checkpoint checkpoints/go2_bc_swing_visible.pt \
  --episodes 5
```

### 4. Evaluate in IsaacLab

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /path/to/simulation/scripts/sim2sim_isaaclab_go2.py --episodes 5
```

### 5. Extract Kick Trajectory for Sim2Real

```bash
cd simulation
conda run --no-capture-output -n mujoco-go2 python scripts/extract_kick_trajectory_mujoco.py
```

Outputs `docs/kick_trajectory.json` with keyframes ready for rl_sar deployment.

## Quick Start — Sim2Real Deployment

The deployed kick is **open-loop keyframe animation**, not a neural-network policy. This is intentional: the MuJoCo/IsaacLab policy uses observations (toe position, ball position, reference trajectory) unavailable on the real robot without external perception.

### 1. Modify rl_sar FSM

Copy `deploy/rl_sar/fsm_robot/fsm_go2.hpp` to your rl_sar repository:

```bash
cp deploy/rl_sar/fsm_robot/fsm_go2.hpp /path/to/rl_sar/src/rl_sar/fsm_robot/fsm_go2.hpp
```

This adds `RLFSMStateSoccerKick` — a new FSM state that reads keyframe parameters from YAML and executes the kick via PD control.

### 2. Install Config

```bash
cp deploy/policy/go2/soccer_kick/config.yaml /path/to/rl_sar/policy/go2/soccer_kick/config.yaml
```

### 3. Build

```bash
cd /path/to/rl_sar
cmake src/rl_sar/ -B cmake_build -DUSE_CMAKE=ON
cmake --build cmake_build --target rl_real_go2 -j4
```

### 4. Run on Real Go2

```bash
./cmake_build/bin/rl_real_go2 enp4s0   # replace with your network interface
```

| Key | Action |
|-----|--------|
| `0` | Stand up / GetUp |
| `2` | SoccerKick (after standing) |
| `P` | Emergency Passive |
| `1` | RL Locomotion |
| `9` | GetDown |

### 5. Safety-First Testing

1. **Start with `safety_scale: 0.5`** in `config.yaml` (half-power)
2. Robot suspended or on flat ground with space in front of FR leg
3. Press `0` to stand → wait → press `2` to kick → press `P` if anything goes wrong
4. After successful tests: increase to `0.75` → `1.0`

## Method Overview

### HRL Architecture

```
Planning Policy (pi_p)          Control Policy (pi_c)
outputs Bezier parameters  ──►  tracks toe trajectory  ──►  12 joint torques
                                 (PPO trained)
```

### Training Pipeline

1. **Stage 1**: Train low-level controller to track randomized kicking toe trajectories (PPO)
2. **Stage 2**: Train high-level planner to output 3x5 Bezier control points (PPO / REDQ)
3. **Behavioral Cloning**: Learn expert keyframe animation for open-loop deployment (supervised learning)
4. **Trajectory Extraction**: Run policy in simulation, record joint angles, extract keyframes at phase boundaries
5. **Sim2Real**: Deploy extracted keyframes as open-loop FSM state on real hardware

### Why Open-Loop Keyframes?

The trained policy needs observations that require external perception (ball position, toe position). The keyframe approach needs only motor encoders and IMU — standard Go2 hardware.

## Key Parameters

See `deploy/policy/go2/soccer_kick/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kick_duration` | 1.00 | Total kick duration (seconds) |
| `safety_scale` | 0.5 | Amplitude scale (0.5=half, 1.0=full) |
| `kick_kp` | 45.0 | PD position stiffness |
| `kick_kd` | 5.0 | PD velocity damping |
| `fr_hip_locked` | true | Lock FR hip to prevent lateral sweep |
| `roll_threshold` | 30.0 | Max roll before attitude protect (degrees) |

## Checkpoints

| File | Method | Description |
|------|--------|-------------|
| `control_policy.pt` | PPO | Low-level toe trajectory tracker |
| `planner_policy.pt` | PPO | High-level Bezier planner |
| `go2_bc_swing_visible.pt` | BC | Behavioral cloning — visible swing (latest) |
| `go2_control_policy.pt` | PPO | Go2 RL control policy |
| `go2_control_policy_v3.pt` | PPO | Go2 RL control policy v3 |

## Sources

- arXiv: https://arxiv.org/abs/2208.01160
- rl_sar: https://github.com/Leoheng22/rl_sar
