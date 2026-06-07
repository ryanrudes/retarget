"""End-to-end capture-to-retarget experiment orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from retarget.observation import SceneObservation
from retarget.pipeline.problem import RetargetingProblem
from retarget.pipeline.recipe import ObservationRecipe, RetargetingRecipe
from retarget.pipeline.retargeter import Retargeter
from retarget.results import RetargetingResult
from retarget.robots.spec import RobotSpec


class RetargetingExperiment:
    """Compose observation processing, robot adaptation, and optimization."""

    def __init__(
        self,
        *,
        observation: ObservationRecipe | SceneObservation,
        recipe: RetargetingRecipe,
        robot: RobotSpec,
        retargeter: Retargeter | None = None,
        retargeter_factory: Callable[[RetargetingProblem], Retargeter] | None = None,
    ) -> None:
        if retargeter is not None and retargeter_factory is not None:
            raise ValueError("use either retargeter or retargeter_factory, not both")
        self._observation_input = observation
        self.recipe = recipe
        self.robot = robot
        self.retargeter = retargeter or Retargeter()
        self.retargeter_factory = retargeter_factory
        self._observed: SceneObservation | None = (
            observation if isinstance(observation, SceneObservation) else None
        )

    def observe(self) -> SceneObservation:
        """Process capture inputs once and retain the in-memory observation."""

        if self._observed is None:
            recipe = cast(ObservationRecipe, self._observation_input)
            self._observed = recipe.observe()
        return self._observed

    def build_problem(self) -> RetargetingProblem:
        """Adapt the observation to the configured robot."""

        return self.recipe.build_problem(self.observe(), self.robot)

    def run(self) -> RetargetingResult:
        """Run the complete in-memory experiment."""

        problem = self.build_problem()
        retargeter = (
            self.retargeter_factory(problem)
            if self.retargeter_factory is not None
            else self.retargeter
        )
        return retargeter.run(problem)
