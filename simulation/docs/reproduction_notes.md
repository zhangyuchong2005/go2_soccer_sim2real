# Reproduction Notes

Paper facts used by this scaffold:

- The robot is Unitree A1 with a floating base and 12 actuated leg joints.
- The shooting maneuver is split into standing, lifting, kicking, and resting phases.
- Low-level control tracks randomized parametric toe trajectories and is optimized with PPO.
- The planner action is Bezier parameters `alpha in R^(3x5)`.
- The planner observation contains target position, ball position history, previous planner output, robot state history, and phase indicator.
- Planner reward is `1.0` within 0.2 m of the goal, otherwise an exponential of squared goal distance.
- The paper uses MuJoCo rigid-ball pretraining and real-world soft-ball fine-tuning.

Implementation choices here:

- `LowLevelTrackingEnv` is a reduced-order latent model, not a full rigid-body A1 model.
- `PlannerBallEnv` uses a calibrated rolling-ball impulse model to make planner experiments fast.
- PPO is used for both stages. The paper used REDQ for the planner; replacing `scripts/train_planner.py` with SAC/REDQ is the main next step for closer reproduction.
- `assets/quadruped_soccer_scene.xml` is a minimal MuJoCo scene for dependency validation and future full-body task work.
