# Extending retarget

Plugin-style extension points keep the core small. Pick the guide that matches what you are adding:

| Guide | When to use it |
|-------|----------------|
| [Add a Robot](adding-a-robot.md) | New `RobotSpec`, kinematics layout, joint limits |
| [Add a Kinematics Backend](adding-a-kinematics-backend.md) | MuJoCo-style FK, Jacobians, integration |
| [Add a Motion Format](adding-a-motion-format.md) | Loaders for new motion file layouts |
| [Add an Objective or Constraint](adding-objectives-constraints.md) | Custom optimization terms |

Protocols and registries are documented in the [API](api/protocols.md) and [Registries](api/registries.md) sections.

Every registry requires a member of its extensible enum-key base. Define a
project enum such as `LabRobot(RobotKind)` or `LabObjective(ObjectiveKind)`,
then pass that member to `register()` and `get()`. Raw strings are accepted only
at config and CLI deserialization boundaries.

Spec and factory registries (`robots`, `motion_formats`,
`kinematics_backends`, `solver_factories`) accept direct values or factories.
Protocol registries (`motion_loaders`, `objective_terms`, `constraint_terms`,
`exporters`, and `visualizers`) validate decorated implementations at
registration time.
