# Add Objectives Or Constraints

Objectives and constraints are small classes implementing `describe()` plus `build(context, spec)`. The default interaction-mesh engine lowers registered objective and constraint specs into quadratic subproblems while keeping the public `RetargetingProblem` schema stable.

Built-in objective names:

- `laplacian`: preserves interaction-mesh local geometry.
- `link_tracking`: tracks named robot links to per-frame target positions stored in `MotionSequence.metadata["link_targets"]`.
- `smoothness`: keeps adjacent frames close in actuated-joint space.
- `nominal_tracking`: tracks configured nominal joints when supplied.

Built-in constraint names:

- `joint_limits`
- `trust_region`
- `foot_contact`
- `foot_lock`
- `non_penetration`: floor and object/terrain sample-point clearance.
- `self_collision`

For research sweeps, compose objective and constraint sets as reusable profiles:

```python
from retarget import OptimizationProfile

profile = (
    OptimizationProfile.defaults(name="low_smoothness")
    .with_objective("smoothness", weight=0.05)
    .with_objective("nominal_tracking", weight=2.0)
    .without_constraint("foot_contact")
    .with_constraint("non_penetration", parameters={"scene_clearance": 0.03})
)

problem = problem.with_optimization_profile(profile)
```

`OptimizationProfile.object_interaction()` and `OptimizationProfile.climbing()` start from the default retargeting profile and add scene non-penetration. Profiles validate against the registered objective, constraint, and solver names with `profile.validate_registry_references()`.

Collision-heavy constraints are intentionally backend-dependent. The fixture backend supports simple ground-style non-penetration and point-distance self-collision checks; MuJoCo-style object and self-collision distances live behind a collision-aware kinematics backend rather than leaking simulator objects into the high-level API.

Self-collision constraints use `KinematicsBackend.collision_candidates()` and then linearize pair separation with `point_jacobians()`:

```python
from retarget.optimization import ConstraintSpec

self_collision = ConstraintSpec(
    name="self_collision",
    parameters={
        "minimum_distance": 0.03,
        "geom_pairs": (("left_hand", "torso"), ("right_hand", "torso")),
    },
)
```

If `geom_pairs` is omitted, the backend chooses its own candidate prefilter. Use explicit pairs for deterministic research experiments and backend defaults for broad simulator collision scans.

Scene non-penetration uses object or terrain `sample_points` and linearizes clearance for selected robot links:

```python
from retarget.optimization import ConstraintSpec

scene_clearance = ConstraintSpec(
    name="non_penetration",
    parameters={
        "scene_clearance": 0.03,
        "activation_distance": 0.08,
        "links": ("left_toe", "right_toe", "left_hand", "right_hand"),
    },
)
```

For dynamic objects, sample points are interpreted in the object frame and robot link positions are converted into that frame before constraints are generated. For terrain, sample points are world-frame points.

CLI run specs can reference scene data from small files instead of embedding arrays inline:

```toml
task_kind = "object_interaction"

[scene.object]
name = "box"
sample_points_path = "box_points.npy"
trajectory_path = "box_trajectory.npz"
```

Point files may be `.npy`, `.npz`, `.json`, or `.csv`. If `sample_points_path` and inline `sample_points` are omitted, `mesh_path` is sampled deterministically with `mesh_sample_count` points. OBJ meshes work without optional dependencies; other mesh formats use the `trimesh` extra. Object trajectory files may use the same suffixes and should provide positions plus optional quaternions; missing quaternions default to identity rotations.

Custom terms use the same typed registries. Objective terms return one or more least-squares contributions over the current actuated-joint increment:

```python
from dataclasses import dataclass

import numpy as np

from retarget.optimization import ObjectiveContribution, ObjectiveSpec, TermContext, objective_terms

@objective_terms.register("energy")
@dataclass(frozen=True)
class EnergyObjective:
    name: str = "energy"

    def describe(self) -> str:
        return "Penalize high-energy joint motion."

    def build(self, context: TermContext, spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]:
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=np.zeros(context.dof, dtype=np.float64),
            ),
        )

```

Then include `ObjectiveSpec(name="energy", weight=0.1)` in a `RetargetingProblem`. The registry stores an instance of the decorated class and validates that it implements the objective protocol. Constraint terms return `ConstraintContribution` with optional joint-increment bounds, a trust radius, and linear constraints.

```python
from dataclasses import dataclass

from retarget.optimization import ConstraintContribution, ConstraintSpec, TermContext, constraint_terms

@constraint_terms.register("my_constraint")
@dataclass(frozen=True)
class MyConstraint:
    name: str = "my_constraint"

    def describe(self) -> str:
        return "Example custom constraint."

    def build(self, context: TermContext, spec: ConstraintSpec) -> ConstraintContribution:
        return ConstraintContribution()
```

Then add `ConstraintSpec(name="my_constraint")` to the problem's constraint list.

For CLI run specs, list extension modules explicitly so they load before registry preflight:

```toml
imports = ["custom_terms.py", "my_lab.retarget_terms"]

[[objectives]]
name = "energy"
weight = 0.1
```

Entries may be dotted module names or `.py` paths relative to the config file.

For new solvers, register a factory that accepts `SolverSpec`:

```python
from retarget.optimization import SolverSpec, solver_factories

@solver_factories.register("my_solver")
def make_solver(spec: SolverSpec) -> object:
    return MySolver(verbose=spec.verbose)
```

`SolverSpec(backend="auto")` prefers the built-in `cvxpy_clarabel` backend when both optional packages are installed, then falls back to `numpy_least_squares` for dependency-light tests and examples. Pin a backend explicitly when comparing optimization methods.
