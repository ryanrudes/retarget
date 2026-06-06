"""Skateboarding ``motion_sync`` source and recipe."""

from retarget.integrations.motion_sync.skateboarding.recipe import SkateboardingRetargetingRecipe
from retarget.integrations.motion_sync.skateboarding.source import SkateboardingClipSource, from_skateboarding_clip
from retarget.integrations.motion_sync.skateboarding.vocabulary import (
    DECK_SAMPLE_POINTS,
    DEFAULT_DEMO,
    FOOT_TARGET_LINKS,
    SkateboardingContactPatch,
    SkateboardingContactState,
    SkateboardingContactSubject,
    SkateboardingDemo,
    SkateboardingGeometryName,
    SkateboardingMotionJoint,
    SkateboardingRobotLink,
)

__all__ = [
    "DECK_SAMPLE_POINTS",
    "DEFAULT_DEMO",
    "FOOT_TARGET_LINKS",
    "SkateboardingClipSource",
    "SkateboardingContactPatch",
    "SkateboardingContactState",
    "SkateboardingContactSubject",
    "SkateboardingDemo",
    "SkateboardingGeometryName",
    "SkateboardingMotionJoint",
    "SkateboardingRetargetingRecipe",
    "SkateboardingRobotLink",
    "from_skateboarding_clip",
]
