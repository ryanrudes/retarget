# Registries

Named plugin registries used by the CLI and `RetargetingProblem` resolution.

Extension registries support direct instances, zero-argument factories, and decorated classes for protocol-backed components. Decorated classes are instantiated immediately and validated against their protocol.

::: retarget.core.registry.Registry

## Built-in registry instances

Each name below is a shared `Registry` instance. Use `.get(name)`, `.register(...)`, and `.names()` in application code. Built-in keys have `StrEnum` aliases such as `Robot.SYNTHETIC_HUMANOID`, `MotionFormat.MINIMAL`, `Objective.LAPLACIAN`, and `Constraint.JOINT_LIMITS`; custom extension keys remain plain strings.

::: retarget.robots.robots

::: retarget.robots.robot_providers

::: retarget.motion.motion_formats

::: retarget.motion.motion_loaders

::: retarget.kinematics.kinematics_backends

::: retarget.optimization.objective_terms

::: retarget.optimization.constraint_terms

::: retarget.optimization.solver_factories

::: retarget.metrics.metrics

::: retarget.export.exporters

::: retarget.visualization.visualizers
