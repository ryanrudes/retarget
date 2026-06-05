"""Define a custom objective descriptor."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from retarget.optimization import (
    ObjectiveConfig,
    ObjectiveContribution,
    OptimizationProfile,
    TermContext,
    objective_terms,
)


class EnergyObjectiveConfig(ObjectiveConfig):
    kind: Literal["energy"] = "energy"


@objective_terms.register("energy")
@dataclass(frozen=True)
class EnergyObjective:
    name: str = "energy"
    config_type: type[EnergyObjectiveConfig] = EnergyObjectiveConfig

    def describe(self) -> str:
        return "Penalize high-energy joint motion."

    def build(self, context: TermContext, _config: EnergyObjectiveConfig) -> tuple[ObjectiveContribution, ...]:
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=np.zeros(context.dof, dtype=np.float64),
            ),
        )


print(objective_terms.get("energy").describe())
profile = OptimizationProfile.defaults().with_objective(EnergyObjectiveConfig(weight=0.1))
print(profile.objective_names)
