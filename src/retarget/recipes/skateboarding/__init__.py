"""Unified capture-to-retarget skateboarding workflow."""

from retarget.recipes.skateboarding.adaptation import SkateboardingRetargetingRecipe
from retarget.recipes.skateboarding.observation import SkateboardingObservationRecipe
from retarget.recipes.skateboarding.schema import GVHMR_SCHEMA, VICON_SCHEMA
from retarget.recipes.skateboarding.vocabulary import (
    DEFAULT_DEMO,
    SkateboardingContactPatch,
    SkateboardingContactState,
    SkateboardingContactSubject,
    SkateboardingDemo,
    SkateboardingGeometryName,
    SkateboardingMotionJoint,
    SkateboardingObservationRole,
    SkateboardingRigidBody,
)

__all__ = [
    "DEFAULT_DEMO",
    "GVHMR_SCHEMA",
    "VICON_SCHEMA",
    "SkateboardingContactPatch",
    "SkateboardingContactState",
    "SkateboardingContactSubject",
    "SkateboardingDemo",
    "SkateboardingGeometryName",
    "SkateboardingMotionJoint",
    "SkateboardingObservationRecipe",
    "SkateboardingObservationRole",
    "SkateboardingRetargetingRecipe",
    "SkateboardingRigidBody",
]
