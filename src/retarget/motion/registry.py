"""Built-in motion registries.

``motion_formats`` maps :class:`~retarget.core.enums.MotionFormat` keys to
:class:`~retarget.motion.spec.MotionFormatSpec` definitions for built-in skeleton
layouts.

``motion_loaders`` maps file suffixes (:class:`~retarget.core.enums.MotionLoaderSuffix`)
to :class:`~retarget.core.protocols.MotionLoader` implementations that parse on-disk
motion files.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from typing import cast

from retarget.core.enums import MotionFormat, MotionFormatKind, MotionJoint, MotionLoaderKind
from retarget.core.protocols import MotionLoader
from retarget.core.registry import Registry
from retarget.motion.spec import MotionFormatSpec


def _motion_format_from_decorator(value: object) -> MotionFormatSpec:
    candidate = value() if callable(value) and not isinstance(value, MotionFormatSpec) else value
    if not isinstance(candidate, MotionFormatSpec):
        raise TypeError("motion format decorators must return MotionFormatSpec")
    return candidate


def _motion_loader_from_decorator(value: object) -> MotionLoader:
    candidate = value
    if isclass(value) or not isinstance(value, MotionLoader):
        if not callable(value):
            raise TypeError("motion loader registrations must implement MotionLoader or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, MotionLoader):
        raise TypeError("motion loader registrations must implement MotionLoader")
    return candidate


motion_formats: Registry[MotionFormatKind, MotionFormatSpec] = Registry(
    "motion format",
    MotionFormatKind,
    decorator_transform=_motion_format_from_decorator,
)
"""Registry of built-in :class:`~retarget.motion.spec.MotionFormatSpec` entries."""

motion_loaders: Registry[MotionLoaderKind, MotionLoader] = Registry(
    "motion loader",
    MotionLoaderKind,
    decorator_transform=_motion_loader_from_decorator,
)
"""Registry of file-suffix :class:`~retarget.core.protocols.MotionLoader` implementations."""


class MinimalMotionJoint(MotionJoint):
    PELVIS = "Pelvis"
    LEFT_HIP = "L_Hip"
    LEFT_KNEE = "L_Knee"
    LEFT_TOE = "L_Toe"
    RIGHT_HIP = "R_Hip"
    RIGHT_KNEE = "R_Knee"
    RIGHT_TOE = "R_Toe"
    SPINE = "Spine"
    HEAD = "Head"
    LEFT_WRIST = "L_Wrist"
    RIGHT_WRIST = "R_Wrist"


class SmplhMotionJoint(MotionJoint):
    PELVIS = "Pelvis"
    LEFT_HIP = "L_Hip"
    LEFT_KNEE = "L_Knee"
    LEFT_ANKLE = "L_Ankle"
    LEFT_TOE = "L_Toe"
    RIGHT_HIP = "R_Hip"
    RIGHT_KNEE = "R_Knee"
    RIGHT_ANKLE = "R_Ankle"
    RIGHT_TOE = "R_Toe"
    TORSO = "Torso"
    SPINE = "Spine"
    CHEST = "Chest"
    NECK = "Neck"
    HEAD = "Head"
    LEFT_SHOULDER = "L_Shoulder"
    LEFT_ELBOW = "L_Elbow"
    LEFT_WRIST = "L_Wrist"
    RIGHT_SHOULDER = "R_Shoulder"
    RIGHT_ELBOW = "R_Elbow"
    RIGHT_WRIST = "R_Wrist"


class LafanMotionJoint(MotionJoint):
    HIPS = "Hips"
    RIGHT_UP_LEG = "RightUpLeg"
    RIGHT_LEG = "RightLeg"
    RIGHT_FOOT = "RightFoot"
    RIGHT_TOE_BASE = "RightToeBase"
    LEFT_UP_LEG = "LeftUpLeg"
    LEFT_LEG = "LeftLeg"
    LEFT_FOOT = "LeftFoot"
    LEFT_TOE_BASE = "LeftToeBase"
    SPINE = "Spine"
    SPINE1 = "Spine1"
    SPINE2 = "Spine2"
    NECK = "Neck"
    HEAD = "Head"
    RIGHT_ARM = "RightArm"
    RIGHT_FORE_ARM = "RightForeArm"
    RIGHT_HAND = "RightHand"
    LEFT_ARM = "LeftArm"
    LEFT_FORE_ARM = "LeftForeArm"
    LEFT_HAND = "LeftHand"


class MocapMotionJoint(MotionJoint):
    HIPS = "Hips"
    SPINE = "Spine"
    SPINE1 = "Spine1"
    NECK = "Neck"
    HEAD = "Head"
    LEFT_ARM = "LeftArm"
    LEFT_FORE_ARM = "LeftForeArm"
    LEFT_HAND = "LeftHand"
    LEFT_HAND_MIDDLE3 = "LeftHandMiddle3"
    RIGHT_ARM = "RightArm"
    RIGHT_FORE_ARM = "RightForeArm"
    RIGHT_HAND = "RightHand"
    RIGHT_HAND_MIDDLE3 = "RightHandMiddle3"
    LEFT_UP_LEG = "LeftUpLeg"
    LEFT_LEG = "LeftLeg"
    LEFT_FOOT = "LeftFoot"
    LEFT_TOE_BASE = "LeftToeBase"
    RIGHT_UP_LEG = "RightUpLeg"
    RIGHT_LEG = "RightLeg"
    RIGHT_FOOT = "RightFoot"
    RIGHT_TOE_BASE = "RightToeBase"


class SmplxMotionJoint(MotionJoint):
    PELVIS = "Pelvis"
    LEFT_HIP = "L_Hip"
    RIGHT_HIP = "R_Hip"
    SPINE1 = "Spine1"
    LEFT_KNEE = "L_Knee"
    RIGHT_KNEE = "R_Knee"
    SPINE2 = "Spine2"
    LEFT_ANKLE = "L_Ankle"
    RIGHT_ANKLE = "R_Ankle"
    SPINE3 = "Spine3"
    LEFT_FOOT = "L_Foot"
    RIGHT_FOOT = "R_Foot"
    NECK = "Neck"
    HEAD = "Head"
    LEFT_SHOULDER = "L_Shoulder"
    RIGHT_SHOULDER = "R_Shoulder"
    LEFT_ELBOW = "L_Elbow"
    RIGHT_ELBOW = "R_Elbow"
    LEFT_WRIST = "L_Wrist"
    RIGHT_WRIST = "R_Wrist"

motion_formats.register(
    MotionFormat.MINIMAL,
    MotionFormatSpec(
        name=MotionFormat.MINIMAL.value,
        joint_vocabulary=MinimalMotionJoint,
        root_joint=MinimalMotionJoint.PELVIS,
        default_height_m=1.7,
        description="Tiny synthetic format used for tests and examples.",
    ),
)
motion_formats.register(
    MotionFormat.SMPLH,
    MotionFormatSpec(
        name=MotionFormat.SMPLH.value,
        joint_vocabulary=SmplhMotionJoint,
        root_joint=SmplhMotionJoint.PELVIS,
        description="Core SMPL-H joints used by common object-interaction datasets.",
    ),
)
motion_formats.register(
    MotionFormat.LAFAN,
    MotionFormatSpec(
        name=MotionFormat.LAFAN.value,
        joint_vocabulary=LafanMotionJoint,
        root_joint=LafanMotionJoint.HIPS,
        default_height_m=1.7,
        description="Core LAFAN-style joint positions.",
    ),
)
motion_formats.register(
    MotionFormat.MOCAP,
    MotionFormatSpec(
        name=MotionFormat.MOCAP.value,
        joint_vocabulary=MocapMotionJoint,
        root_joint=MocapMotionJoint.HIPS,
        default_height_m=1.78,
        description="Core markerless mocap-style joint positions.",
    ),
)
motion_formats.register(
    MotionFormat.SMPLX,
    MotionFormatSpec(
        name=MotionFormat.SMPLX.value,
        joint_vocabulary=SmplxMotionJoint,
        root_joint=SmplxMotionJoint.PELVIS,
        description="Core SMPL-X global joint position format.",
    ),
)
