"""Define a custom objective descriptor."""

from dataclasses import dataclass

import numpy as np

from retarget.optimization import ObjectiveContribution, ObjectiveSpec, TermContext, objective_terms


@objective_terms.register("energy")
@dataclass(frozen=True)
class EnergyObjective:
    name: str = "energy"
    weight: float = 0.1

    def describe(self) -> str:
        return "Penalize high-energy joint motion."

    def build(self, context: TermContext, _spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]:
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=np.zeros(context.dof, dtype=np.float64),
            ),
        )


print(objective_terms.get("energy").describe())
