# Add Objectives Or Constraints

Objectives and constraints are registered terms paired with typed config models. A term exposes `config_type`, and `build(context, config)` receives that model directly. Config tables and Python code use concrete fields, not generic parameter dictionaries.

Built-in objectives:

- `laplacian`: preserves interaction-mesh local geometry.
- `link_tracking`: tracks robot links to a `LinkTargetPlan`.
- `smoothness`: keeps adjacent frames close in actuated-joint space.
- `nominal_tracking`: tracks configured nominal joints.

Built-in constraints:

- `joint_limits`
- `trust_region`
- `foot_sticking`
- `foot_lock`
- `non_penetration`: support-plane, object/terrain sample-point, or backend geometry clearance.
- `self_collision`

For research sweeps, compose reusable profiles with config objects:

```python
from retarget.optimization import (
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    OptimizationProfile,
    SmoothnessObjectiveConfig,
)

profile = (
    OptimizationProfile.defaults(name="low_smoothness")
    .with_objective(SmoothnessObjectiveConfig(weight=0.05))
    .with_objective(NominalTrackingObjectiveConfig(weight=2.0))
    .without_constraint("foot_sticking")
    .with_constraint(NonPenetrationConstraintConfig(scene_clearance=0.03))
)

problem = problem.with_optimization_profile(profile)
```

`OptimizationProfile.object_interaction()` and `OptimizationProfile.climbing()` start from the default profile and add scene non-penetration. Profiles validate against registered objective, constraint, and solver names with `profile.validate_registry_references()`.

Self-collision constraints use `KinematicsBackend.collision_candidates()` and then linearize pair separation with `point_jacobians()`:

```python
from retarget.optimization import SelfCollisionConstraintConfig

self_collision = SelfCollisionConstraintConfig(
    minimum_distance=0.03,
    pairs=(("left_hand", "torso"), ("right_hand", "torso")),
)
```

If `pairs` is omitted, the backend chooses its own candidate prefilter. Use explicit pairs for deterministic research experiments and backend defaults for broad simulator collision scans.

Scene non-penetration uses object or terrain `sample_points` and linearizes clearance for selected robot links:

```python
from retarget.optimization import NonPenetrationConstraintConfig

scene_clearance = NonPenetrationConstraintConfig(
    sources=("support", "scene_points"),
    scene_clearance=0.03,
    activation_distance=0.08,
    links=("left_toe", "right_toe", "left_hand", "right_hand"),
)
```

For dynamic objects, sample points are interpreted in the object frame and robot link positions are converted into that frame before constraints are generated. For terrain, sample points are world-frame points.

`sources` controls which linearizations are active. Use `("support",)` for floor/support-plane clearance, `("scene_points",)` for sampled object or terrain clearance, and `("geometry",)` for backend signed-distance constraints such as MuJoCo geom pairs. Compatibility profiles can leave `non_penetration` out entirely when a reference system disables collision constraints.

Custom objective terms define a config class and register a term that consumes it:

```python
from dataclasses import dataclass
from typing import Literal

import numpy as np

from retarget.optimization import ObjectiveConfig, ObjectiveContribution, TermContext, objective_terms


class EnergyObjectiveConfig(ObjectiveConfig):
    kind: Literal["energy"] = "energy"


@objective_terms.register("energy")
@dataclass(frozen=True)
class EnergyObjective:
    name: str = "energy"
    config_type: type[EnergyObjectiveConfig] = EnergyObjectiveConfig

    def describe(self) -> str:
        return "Penalize high-energy joint motion."

    def build(self, context: TermContext, config: EnergyObjectiveConfig) -> tuple[ObjectiveContribution, ...]:
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=np.zeros(context.dof, dtype=np.float64),
            ),
        )
```

Then include `EnergyObjectiveConfig(weight=0.1)` in a `RetargetingProblem`, or use `kind = "energy"` in a run config after importing the extension. Constraint terms follow the same pattern:

```python
from dataclasses import dataclass
from typing import Literal

from retarget.optimization import ConstraintConfig, ConstraintContribution, TermContext, constraint_terms


class MyConstraintConfig(ConstraintConfig):
    kind: Literal["my_constraint"] = "my_constraint"


@constraint_terms.register("my_constraint")
@dataclass(frozen=True)
class MyConstraint:
    name: str = "my_constraint"
    config_type: type[MyConstraintConfig] = MyConstraintConfig

    def describe(self) -> str:
        return "Example custom constraint."

    def build(self, context: TermContext, config: MyConstraintConfig) -> ConstraintContribution:
        return ConstraintContribution()
```

For CLI run specs, list extension modules explicitly so they load before registry preflight:

```toml
imports = ["custom_terms.py", "my_lab.retarget_terms"]

[[objectives]]
kind = "energy"
weight = 0.1
```

Entries may be dotted module names or `.py` paths relative to the config file. The loader validates each table through the registered term's `config_type`, so bad fields fail before the run starts.

For new solvers, register a factory that accepts `SolverSpec`:

```python
from retarget.optimization import SolverSpec, solver_factories

@solver_factories.register("my_solver")
def make_solver(spec: SolverSpec) -> object:
    return MySolver(verbose=spec.verbose)
```

`SolverSpec(backend="auto")` prefers the built-in `cvxpy_clarabel` backend when both optional packages are installed, then falls back to `numpy_least_squares` for dependency-light tests and examples. Pin a backend explicitly when comparing optimization methods.
