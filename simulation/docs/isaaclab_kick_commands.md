# IsaacLab Go2 Kick Commands

This file records the commands for reproducing the IsaacLab Go2 soccer-kick demo.
It only runs sim2sim in IsaacLab. It does not deploy to the real robot.

## Paths

```bash
PROJECT=/home/yczhang/hrl_soccer_shooting
ISAACLAB=/home/yczhang/IsaacLab-2.1.0
CHECKPOINT=/home/yczhang/hrl_soccer_shooting/checkpoints/go2_bc_swing_visible.pt
```

Conda environment:

```bash
isaaclab.
```

The trailing dot is part of the environment name.

## Default Visual Run

This is the main command to watch the current tuned kick:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --episodes 1 \
  --horizon 220 \
  --hold-after 5
```

Current defaults in the script:

```text
checkpoint=checkpoints/go2_bc_swing_visible.pt
controller=policy
ball-x=0.44
ball-y=-0.150
ball-z=0.105
ordered-swing=1.0
center-hit-lift=0.20
skill-speed=4.0
asset-source=local-urdf
fixed-base
```

Behavior:

- The front-right leg backswing happens before the shot.
- The kick phase runs about 4 times faster than the original slow preview.
- The ball is close to the Go2.
- The foot hits near the ball center instead of the bottom.

## Headless Verification

Use this when you only want metrics:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --headless \
  --episodes 1 \
  --horizon 220 \
  --progress-every 220
```

Expected metric format:

```text
episode=01 ball_move=... goal_dist=... min_base_h=... closest_toe_z=... impact_toe_z=... impact_phase=... ball_z=...
```

Reference result from the tuned 4x run:

```text
episode=01 ball_move=5.043 goal_dist=3.196 min_base_h=0.290 closest_toe_z=0.077 impact_toe_z=0.085 impact_phase=0.47 ball_z=0.105
```

Interpretation:

- `impact_phase > 0.30`: backswing happened before ball contact.
- `impact_toe_z` close to `ball_z`: hit point is near the ball center height.
- `ball_move > 0`: ball was kicked.

## Speed Tuning

Default is 4x:

```bash
--skill-speed 4.0
```

Slower, easier to inspect:

```bash
--skill-speed 3.0
```

Faster, sharper kick:

```bash
--skill-speed 5.0
```

Example:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --episodes 1 \
  --horizon 220 \
  --skill-speed 5.0 \
  --hold-after 5
```

## Ball Distance Tuning

Current default:

```bash
--ball-x 0.44
```

Closer to the robot:

```bash
--ball-x 0.42
```

Farther from the robot:

```bash
--ball-x 0.46
```

Notes:

- Smaller `ball-x` means the ball is closer.
- If the ball is too close, the foot can hit too early or too low.
- If the ball is too far, the kick can look cleaner but less close to the Go2.

## Hit Height Tuning

Current default:

```bash
--center-hit-lift 0.20
```

Lower contact, usually stronger:

```bash
--center-hit-lift 0.0
```

Higher contact, usually weaker:

```bash
--center-hit-lift 0.40
```

Use `impact_toe_z` and `ball_z` from the headless output to tune this. For example,
if `impact_toe_z` is much smaller than `ball_z`, increase `--center-hit-lift`.

## Backswing Timing

The explicit timing guard is enabled by default:

```bash
--ordered-swing 1.0
```

It forces:

```text
phase 0.00-0.30: backswing
phase 0.30-0.56: forward strike
phase 0.56-0.72: follow-through
```

Disable it only for debugging the raw policy:

```bash
--ordered-swing 0.0
```

## Raw Policy Debug Run

This shows the checkpoint behavior without the explicit ordered swing guard:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --episodes 1 \
  --horizon 220 \
  --ordered-swing 0.0 \
  --center-hit-lift 0.0 \
  --skill-speed 1.0 \
  --hold-after 5
```

## Expert Controller Debug Run

This bypasses the learned policy and uses the scripted controller:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --episodes 1 \
  --controller expert \
  --horizon 220 \
  --hold-after 5
```

## Quick Smoke Test

Use this only to check whether the script launches:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --headless \
  --episodes 1 \
  --horizon 5 \
  --settle-steps 0 \
  --progress-every 1
```

## Common Variants

Closer ball, keep the ordered 4x swing:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --episodes 1 \
  --horizon 220 \
  --ball-x 0.42 \
  --hold-after 5
```

More center-height contact:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --episodes 1 \
  --horizon 220 \
  --center-hit-lift 0.30 \
  --hold-after 5
```

Slower visual inspection:

```bash
cd /home/yczhang/IsaacLab-2.1.0
TERM=xterm conda run --no-capture-output -n isaaclab. ./isaaclab.sh \
  -p /home/yczhang/hrl_soccer_shooting/scripts/sim2sim_isaaclab_go2.py \
  --episodes 1 \
  --horizon 220 \
  --skill-speed 3.0 \
  --hold-after 5
```

## Troubleshooting

If the dog does not kick:

- Check the checkpoint path:
  `/home/yczhang/hrl_soccer_shooting/checkpoints/go2_bc_swing_visible.pt`
- Run the headless verification command and check `ball_move`.
- Keep `--asset-source local-urdf`; this path preserves the `FR_foot` body.

If backswing appears after the ball moves:

- Keep `--ordered-swing 1.0`.
- Check that `impact_phase` is greater than `0.30`.

If the foot hits the ball bottom:

- Increase `--center-hit-lift`, for example `--center-hit-lift 0.30`.
- Or move the ball slightly farther, for example `--ball-x 0.46`.

If the foot misses high:

- Decrease `--center-hit-lift`, for example `--center-hit-lift 0.0`.
- Or move the ball slightly closer, for example `--ball-x 0.42`.
