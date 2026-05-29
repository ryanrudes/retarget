# Troubleshooting

Run `retarget doctor` to see registered motion formats, loaders, robots, kinematics backends, objective terms, constraint terms, solver factories, metrics, exporters, visualizers, and optional dependency availability. Install feature extras only when needed, for example `uv sync --extra optimize --extra mujoco`.

Run configs are preflighted against the extension registries before motions or assets are loaded. If a CLI run reports an unknown objective, constraint, solver, robot, or motion format, add the extension module to the run spec's `imports` list or import/register it in the process that builds the `RetargetingProblem`, then rerun `retarget doctor` to confirm the name appears.
