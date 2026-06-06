"""Define a custom objective descriptor."""

from dataclasses import dataclass

import numpy as np

from retarget import RetargetEnum
from retarget.optimization import (
    ObjectiveConfig,
    ObjectiveContribution,
    OptimizationProfile,
    TermContext,
    objective_terms,
)


class DemoObjective(RetargetEnum):
    ENERGY = "energy"


class EnergyObjectiveConfig(ObjectiveConfig):
    kind: DemoObjective = DemoObjective.ENERGY


@objective_terms.register(DemoObjective.ENERGY)
@dataclass(frozen=True)
class EnergyObjective:
    name: DemoObjective = DemoObjective.ENERGY
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


print(objective_terms.get(DemoObjective.ENERGY).describe())
profile = OptimizationProfile.defaults().with_objective(EnergyObjectiveConfig(weight=0.1))
print(profile.objective_names)
