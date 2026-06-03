# Extending retarget

Plugin-style extension points keep the core small. Pick the guide that matches what you are adding:

| Guide | When to use it |
|-------|----------------|
| [Add a Robot](adding-a-robot.md) | New `RobotSpec`, kinematics layout, joint limits |
| [Add a Kinematics Backend](adding-a-kinematics-backend.md) | MuJoCo-style FK, Jacobians, integration |
| [Add a Motion Format](adding-a-motion-format.md) | Loaders for new motion file layouts |
| [Add an Objective or Constraint](adding-objectives-constraints.md) | Custom optimization terms |

Protocols and registries are documented in the [API](api/protocols.md) and [Registries](api/registries.md) sections.

Registries accept direct values, zero-argument factories, or decorated classes. **Spec and factory registries** (`robots`, `motion_formats`, `kinematics_backends`, `solver_factories`) usually register `@registry.register("name")` on a function that returns a `RobotSpec`, `MotionFormatSpec`, `KinematicsBackendFactory`, or `SolverFactory`. **Protocol registries** (`motion_loaders`, `objective_terms`, `constraint_terms`, `exporters`, `visualizers`, and similar) instantiate decorated classes at registration time and validate them against their protocol so missing methods fail early.
