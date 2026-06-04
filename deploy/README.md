# Sim2Real Deployment — rl_sar SoccerKick

This directory contains everything needed to deploy the Go2 soccer kick on real hardware.

> **Note**: `rl_sar` is a third-party project ([Leoheng22/rl_sar](https://github.com/Leoheng22/rl_sar)). This repo only contains our **modifications** (FSM state) and **config files**, not the full rl_sar source code.

## Files

| File | Description |
|------|-------------|
| `rl_sar/fsm_robot/fsm_go2.hpp` | Modified Go2 FSM with `RLFSMStateSoccerKick` state |
| `policy/go2/soccer_kick/config.yaml` | Soccer kick keyframe parameters (extracted from simulation) |
| `install_soccer_kick_to_rl_sar.sh` | Install script — copies config + builds rl_real_go2 |
| `sim2real_with_rl_sar.md` | Full deployment guide with safety guide and tuning |

## SoccerKick FSM State

The `RLFSMStateSoccerKick` state executes an **open-loop keyframe animation** of the FR leg kicking motion:

```
Phase 0.00 → 0.35:  Backswing (FR leg swings backward)
Phase 0.35 → 0.65:  Strike (FR leg sweeps forward through ball)
Phase 0.65 → 0.80:  Follow-through
Phase 0.80 → 1.00:  Return to standing pose
```

### Keyframes (SDK order: FR_hip, FR_thigh, FR_calf)

These were extracted from the `go2_bc_swing_visible.pt` policy running in MuJoCo simulation:

| Keyframe | Value |
|----------|-------|
| `fr_backswing` | [0.00, 1.60, -2.85] |
| `fr_strike`    | [-0.10, -0.04, -1.63] |
| `fr_follow`    | [-0.16, -0.55, -1.15] |

### Safety Features

- **AttitudeProtect**: Every step checks roll/pitch against thresholds. If exceeded → immediate passive mode (kick aborted).
- **safety_scale**: Scales deviation from standing pose. Start at 0.5, increase gradually.
- **fr_hip_locked**: Keeps FR hip at standing angle to prevent lateral sweep.

## Quick Start

### 1. Clone rl_sar

```bash
git clone https://github.com/Leoheng22/rl_sar.git
cd rl_sar
```

### 2. Apply our modifications

```bash
# Option A: Use install script
bash /path/to/go2_soccer_sim2real/deploy/install_soccer_kick_to_rl_sar.sh

# Option B: Manual
cp /path/to/go2_soccer_sim2real/deploy/rl_sar/fsm_robot/fsm_go2.hpp src/rl_sar/fsm_robot/
cp /path/to/go2_soccer_sim2real/deploy/policy/go2/soccer_kick/config.yaml policy/go2/soccer_kick/
```

### 3. Build rl_real_go2

```bash
cmake src/rl_sar/ -B cmake_build -DUSE_CMAKE=ON
cmake --build cmake_build --target rl_real_go2 -j4
```

### 4. Run

```bash
./cmake_build/bin/rl_real_go2 enp4s0   # replace with your network interface
```

| Key | Action |
|-----|--------|
| `0` | GetUp (stand up) |
| `2` | **SoccerKick** (after standing) |
| `P` | Emergency Passive |
| `1` | RL Locomotion |
| `9` | GetDown |

### 5. Safety

1. **Start with `safety_scale: 0.5`** in `config.yaml`
2. Test on flat ground, clear space in front of FR leg
3. Press `P` immediately if robot leans or behaves unexpectedly

See `sim2real_with_rl_sar.md` for the full safety guide and tuning instructions.
