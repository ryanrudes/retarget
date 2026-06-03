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

from retarget.core.enums import MotionFormat
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


motion_formats: Registry[MotionFormatSpec] = Registry(
    "motion format",
    decorator_transform=_motion_format_from_decorator,
)
"""Registry of built-in :class:`~retarget.motion.spec.MotionFormatSpec` entries."""

motion_loaders: Registry[MotionLoader] = Registry(
    "motion loader",
    decorator_transform=_motion_loader_from_decorator,
)
"""Registry of file-suffix :class:`~retarget.core.protocols.MotionLoader` implementations."""


MINIMAL_JOINTS = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Toe",
    "Spine",
    "Head",
    "L_Wrist",
    "R_Wrist",
)

SMPLH_CORE_JOINTS = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
)

LAFAN_CORE_JOINTS = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)

MOCAP_CORE_JOINTS = (
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandMiddle3",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandMiddle3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)

SMPLX_CORE_JOINTS = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
)

motion_formats.register(
    MotionFormat.MINIMAL,
    MotionFormatSpec(
        name=MotionFormat.MINIMAL.value,
        joint_names=MINIMAL_JOINTS,
        root_joint="Pelvis",
        contact_joints=("L_Toe", "R_Toe"),
        default_height_m=1.7,
        description="Tiny synthetic format used for tests and examples.",
    ),
)
motion_formats.register(
    MotionFormat.SMPLH,
    MotionFormatSpec(
        name=MotionFormat.SMPLH.value,
        joint_names=SMPLH_CORE_JOINTS,
        root_joint="Pelvis",
        contact_joints=("L_Toe", "R_Toe"),
        description="Core SMPL-H joints used by common object-interaction datasets.",
    ),
)
motion_formats.register(
    MotionFormat.LAFAN,
    MotionFormatSpec(
        name=MotionFormat.LAFAN.value,
        joint_names=LAFAN_CORE_JOINTS,
        root_joint="Hips",
        contact_joints=("LeftToeBase", "RightToeBase"),
        default_height_m=1.7,
        description="Core LAFAN-style joint positions.",
    ),
)
motion_formats.register(
    MotionFormat.MOCAP,
    MotionFormatSpec(
        name=MotionFormat.MOCAP.value,
        joint_names=MOCAP_CORE_JOINTS,
        root_joint="Hips",
        contact_joints=("LeftToeBase", "RightToeBase"),
        default_height_m=1.78,
        description="Core markerless mocap-style joint positions.",
    ),
)
motion_formats.register(
    MotionFormat.SMPLX,
    MotionFormatSpec(
        name=MotionFormat.SMPLX.value,
        joint_names=SMPLX_CORE_JOINTS,
        root_joint="Pelvis",
        contact_joints=("L_Foot", "R_Foot"),
        description="Core SMPL-X global joint position format.",
    ),
)
