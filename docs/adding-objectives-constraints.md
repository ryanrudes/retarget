# Add Objectives Or Constraints

Objective and constraint config classes declare a typed class-level `kind`.
Registries are keyed by members of extensible enum bases.

```python
from dataclasses import dataclass

import numpy as np

from retarget import ObjectiveKind
from retarget.optimization import (
    ObjectiveConfig,
    ObjectiveContribution,
    TermContext,
    objective_terms,
)


class LabObjective(ObjectiveKind):
    ENERGY = "energy"


class EnergyObjectiveConfig(ObjectiveConfig):
    kind = LabObjective.ENERGY


@objective_terms.register(LabObjective.ENERGY)
@dataclass(frozen=True)
class EnergyObjective:
    kind: LabObjective = LabObjective.ENERGY
    config_type: type[EnergyObjectiveConfig] = EnergyObjectiveConfig

    def describe(self) -> str:
        return "Penalize high-energy joint motion."

    def build(
        self,
        context: TermContext,
        config: EnergyObjectiveConfig,
    ) -> tuple[ObjectiveContribution, ...]:
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof),
                target=np.zeros(context.dof),
            ),
        )
```

Compose typed reusable profiles:

```python
from retarget import Constraint
from retarget.optimization import (
    NominalTrackingObjectiveConfig,
    OptimizationProfile,
    SmoothnessObjectiveConfig,
)

profile = (
    OptimizationProfile.defaults(name="low_smoothness")
    .with_objective(SmoothnessObjectiveConfig(weight=0.05))
    .with_objective(NominalTrackingObjectiveConfig(joints=nominal_joints))
    .without_constraint(Constraint.FOOT_STICKING)
)
```

Collision constraints require explicit typed selections:

```python
from retarget import NonPenetrationSource
from retarget.optimization import GeometryPair, NonPenetrationConstraintConfig

clearance = NonPenetrationConstraintConfig(
    sources=(NonPenetrationSource.GEOMETRY,),
    geometry_pairs=(
        GeometryPair(
            first=LabGeometry.LEFT_FOOT,
            second=LabSceneGeometry.GROUND,
        ),
    ),
)
```

There is no backend-selected all-pairs fallback. Missing links, subjects, or
geometry pairs fail before optimization.

Declarative configs may use `kind = "energy"` only after importing the module
that registers `LabObjective.ENERGY`. Config preflight resolves the string to
that concrete enum member and rejects unknown registrations.

Solver extensions follow the same pattern with `SolverKind`:

```python
from retarget import SolverKind
from retarget.optimization import SolverSpec, solver_factories


class LabSolver(SolverKind):
    CUSTOM = "lab_solver"


@solver_factories.register(LabSolver.CUSTOM)
def make_solver(spec: SolverSpec) -> object:
    return MySolver(verbose=spec.verbose)
```
