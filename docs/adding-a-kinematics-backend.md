# Add A Kinematics Backend

Kinematics backends implement `KinematicsBackend` and are registered in `kinematics_backends`. The retargeting engine uses this protocol instead of depending on a simulator directly.

Required operations:

- `forward_kinematics` and `link_positions` for named body/link positions
- `body_jacobians` for translational and rotational Jacobians
- `point_jacobians` for point positions and translational Jacobians used by the optimizer
- `qpos_to_qvel` and `integrate_qvel` for simulator-safe qpos/qvel conversion
- `joint_limits` for actuator bounds used directly by the optimizer
- `geom_distances` for explicit geometry pairs used by collision-aware constraints

Register a backend factory with a typed key:

```python
from retarget import KinematicsKind
from retarget.kinematics import kinematics_backends
from retarget.pipeline.compiled import CompiledRobotSpec


class LabKinematics(KinematicsKind):
    BACKEND = "lab_backend"


@kinematics_backends.register(LabKinematics.BACKEND)
def make_backend(robot: CompiledRobotSpec) -> MyKinematicsBackend:
    return MyKinematicsBackend(robot)
```

The built-in `SimpleKinematicsBackend` is deterministic and dependency-light for tests. `MuJoCoKinematicsBackend` keeps MuJoCo-specific FK, Jacobians, position integration, joint range extraction, and geom-distance calls behind the same protocol.

Backend joint limits are the optimizer source of truth. MuJoCo extracts limited hinge and slide ranges by exact joint name or by qpos order after the configured root layout; explicit `RobotSpec.joint_limits` then override those ranges for experiments that need tighter or corrected bounds.
