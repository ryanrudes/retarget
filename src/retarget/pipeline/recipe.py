"""Typed observation and robot-adaptation recipe protocols."""

from __future__ import annotations

from typing import Protocol

from retarget.observation import SceneObservation
from retarget.pipeline.problem import RetargetingProblem
from retarget.robots.spec import RobotSpec
from retarget.scene.spec import SceneSpec


class ObservationRecipe(Protocol):
    """Process native recordings into a target-independent observation."""

    def observe(self) -> SceneObservation:
        """Return a shared-timeline scene observation."""


class SceneRecipe(Protocol):
    """Build target-independent runtime scene data from an observation."""

    def build_scene(self, observation: SceneObservation) -> SceneSpec:
        """Build a frame-aligned scene."""


class RetargetingRecipe(Protocol):
    """Adapt a scene observation to one target robot."""

    def build_problem(
        self,
        observation: SceneObservation,
        robot: RobotSpec,
    ) -> RetargetingProblem:
        """Build a complete retargeting problem."""
