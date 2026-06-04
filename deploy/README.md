# Sim2Real Deployment — rl_sar SoccerKick

This directory contains the modified rl_sar FSM state for deploying the Go2 soccer kick on real hardware.

## Files

| File | Description |
|------|-------------|
| `rl_sar/fsm_robot/fsm_go2.hpp` | Modified Go2 FSM with `RLFSMStateSoccerKick` state |
| `policy/go2/soccer_kick/config.yaml` | Soccer kick keyframe parameters (extracted from simulation) |

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

### Integration Steps

1. Replace `fsm_go2.hpp` in your rl_sar repo
2. Copy `config.yaml` to `rl_sar/policy/go2/soccer_kick/`
3. Rebuild `rl_real_go2`
4. Connect to Go2 via ethernet
5. Run `./cmake_build/bin/rl_real_go2 <interface>`
