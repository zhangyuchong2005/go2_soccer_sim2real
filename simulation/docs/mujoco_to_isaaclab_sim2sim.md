# MuJoCo -> IsaacLab Go2 Sim2Sim

This stage only validates the MuJoCo-trained kicking policy inside IsaacLab simulation.
It does not touch `rl_sar` or the real Go2.

## What Is Reused

- MuJoCo checkpoint: `checkpoints/go2_bc_swing_visible.pt`
- MuJoCo observation layout:
  - 12 joint positions
  - 12 joint velocities
  - base position
  - front-right toe position
  - toe reference position
  - ball xy
  - phase
- MuJoCo normalized action layout:
  - 12 joint residuals in `FL, FR, RL, RR` order

## What Is New

`scripts/sim2sim_isaaclab_go2.py` launches IsaacLab, spawns:

- Unitree Go2 converted from the local `unitree_rl_gym` URDF by default
- A dynamic soccer ball sphere
- A ground plane

The script maps MuJoCo joint order to IsaacLab joint ids by name, reconstructs the
MuJoCo observation from IsaacLab state tensors, and applies the MuJoCo policy action
as IsaacLab joint position targets.

The default is fixed-base because the MuJoCo BC/evaluation path used a pinned-base
preview. This validates the kick transfer first. Use `--floating-base` later for
balance testing after the fixed-base action/ball contact looks right.

## Run

From the IsaacLab folder:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
    -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
    --headless \
    --episodes 3 \
    --checkpoint /home/yczhang/hrl_soccer_shooting/checkpoints/go2_bc_swing_visible.pt
```

Conservative first pass:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
    -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
    --headless \
    --episodes 3 \
    --action-scale 0.5
```

Headless metric run:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
    -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
    --headless \
    --episodes 10
```

Quick smoke test:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
    -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
    --headless \
    --episodes 1 \
    --horizon 5 \
    --settle-steps 0 \
    --action-scale 0.5 \
    --progress-every 1
```

## Expected Output

Each episode prints:

```text
episode=01 ball_move=... goal_dist=... min_base_h=... closest_toe_z=... impact_toe_z=... impact_phase=... ball_z=...
```

Verified locally:

```text
episode=01 ball_move=... goal_dist=... min_base_h=... closest_toe_z=... impact_toe_z=... impact_phase=... ball_z=...
```

That result used `--headless --episodes 1 --controller policy --action-scale 1.0`
with the default fixed-base URDF conversion and `go2_bc_swing_visible.pt`.

The default ball position is now `--ball-x 0.44` with `--ordered-swing 1.0`,
`--center-hit-lift 0.20`, and `--skill-speed 4.0`. The ordered swing forces the
front-right leg to move backward first, then forward into the ball, so the visible
backswing happens before the shot. The skill speed compresses the kick to about four
times the original duration.

Useful tuning values:

- `--ball-x 0.42`: closer to the robot, but more likely to hit low.
- `--ball-x 0.44`: default close placement.
- `--center-hit-lift 0.0`: lower contact, stronger hit.
- `--center-hit-lift 0.20`: default center-ish contact.
- `--center-hit-lift 0.40`: higher contact, weaker hit.
- `--ordered-swing 0.0`: disable the explicit backswing-before-strike timing guard.
- `--skill-speed 3.0`: slower, easier to inspect.
- `--skill-speed 5.0`: faster, sharper kick.

Verified default:

```text
episode=01 ball_move=5.043 goal_dist=3.196 min_base_h=0.290 closest_toe_z=0.077 impact_toe_z=0.085 impact_phase=0.47 ball_z=0.105
```

If `ball_move` is near zero in IsaacLab while MuJoCo works, the next debugging target is
physics/action mismatch, not training. In that case compare:

- joint sign convention
- foot body used as toe proxy
- default pose
- action scale
- PD stiffness/damping
- ball location and radius

## Notes

- `--asset-source local-urdf` is the default and is the path verified locally.
- `--asset-source local-mjcf` currently converts, but Isaac Sim 4.5 creates duplicate
  articulation roots for this MJCF, so it is not the default.
- `--asset-source isaaclab-usd` depends on IsaacLab/Nucleus Go2 USD access. On this
  machine, Nucleus/OmniHub was inaccessible during testing.
