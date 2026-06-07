"""Typed fixtures shared by domain and compilation tests."""

from __future__ import annotations

import numpy as np

from retarget.capture import SampleTimeline
from retarget.core.enums import (
    ContactPatch,
    ContactState,
    ContactSubject,
    FrameConvention,
    MotionJoint,
    RobotGeometry,
    RobotJoint,
    RobotLink,
    RobotRole,
    SolverBackend,
    TaskKind,
)
from retarget.motion import ContactPlan, ContactTrack, LinkTargetPlan, LinkTargetTrack
from retarget.motion.spec import MotionSequence
from retarget.optimization import (
    JointLimitsConstraintConfig,
    SolverSpec,
    TrustRegionConstraintConfig,
)
from retarget.pipeline import LinkBinding, RetargetingProblem
from retarget.robots import RobotSpec, RobotVocabulary, SimpleKinematicPoint
from retarget.scene import SceneSpec


class FixtureMotionJoint(MotionJoint):
    ROOT = "root"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class SameValueMotionJoint(MotionJoint):
    ROOT = "root"


class FixtureRobotJoint(RobotJoint):
    ROOT = "root_joint"
    LEFT_LEG = "left_leg_joint"
    RIGHT_LEG = "right_leg_joint"


class SameValueRobotJoint(RobotJoint):
    ROOT = "root_joint"


class FixtureRobotLink(RobotLink):
    PELVIS = "pelvis"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class SameValueRobotLink(RobotLink):
    PELVIS = "pelvis"


class FixtureRobotGeometry(RobotGeometry):
    LEFT_FOOT = "left_foot_geom"
    RIGHT_FOOT = "right_foot_geom"


class SameValueRobotGeometry(RobotGeometry):
    LEFT_FOOT = "left_foot_geom"


class FixtureRobotRole(RobotRole):
    PELVIS = "pelvis"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class FixtureContactSubject(ContactSubject):
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class SameValueContactSubject(ContactSubject):
    LEFT_FOOT = "left_foot"


class FixtureContactState(ContactState):
    AIR = "air"
    SUPPORT = "support"


class FixtureContactPatch(ContactPatch):
    LEFT_SOLE = "left_sole"
    RIGHT_SOLE = "right_sole"


def fixture_motion(frame_count: int = 3) -> MotionSequence[FixtureMotionJoint]:
    timeline = SampleTimeline.uniform(frame_count, 30.0, clock="fixture")
    positions = np.zeros((frame_count, len(FixtureMotionJoint), 3), dtype=np.float64)
    positions[:, 1] = (-0.15, 0.1, -0.8)
    positions[:, 2] = (0.15, 0.1, -0.8)
    positions[:, :, 0] += np.arange(frame_count, dtype=np.float64)[:, None] * 0.01
    return MotionSequence(
        name="fixture_motion",
        joint_vocabulary=FixtureMotionJoint,
        joints=tuple(FixtureMotionJoint),
        root_joint=FixtureMotionJoint.ROOT,
        joint_positions=positions,
        timeline=timeline,
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
        source_height_m=1.7,
        provenance={"source": "unit_test"},
    )


def fixture_robot() -> RobotSpec[
    FixtureRobotJoint,
    FixtureRobotLink,
    FixtureRobotGeometry,
    FixtureRobotRole,
]:
    return RobotSpec(
        name="fixture_robot",
        height_m=1.0,
        vocabulary=RobotVocabulary(
            joints=FixtureRobotJoint,
            links=FixtureRobotLink,
            geometries=FixtureRobotGeometry,
            roles=FixtureRobotRole,
        ),
        joints=tuple(FixtureRobotJoint),
        links=tuple(FixtureRobotLink),
        contact_links=(FixtureRobotLink.LEFT_FOOT, FixtureRobotLink.RIGHT_FOOT),
        geometries=tuple(FixtureRobotGeometry),
        joint_limits={joint: (-1.0, 1.0) for joint in FixtureRobotJoint},
        joint_roles={
            FixtureRobotRole.PELVIS: FixtureRobotJoint.ROOT,
            FixtureRobotRole.LEFT_FOOT: FixtureRobotJoint.LEFT_LEG,
            FixtureRobotRole.RIGHT_FOOT: FixtureRobotJoint.RIGHT_LEG,
        },
        link_roles={
            FixtureRobotRole.PELVIS: FixtureRobotLink.PELVIS,
            FixtureRobotRole.LEFT_FOOT: FixtureRobotLink.LEFT_FOOT,
            FixtureRobotRole.RIGHT_FOOT: FixtureRobotLink.RIGHT_FOOT,
        },
        simple_kinematics={
            FixtureRobotLink.PELVIS: SimpleKinematicPoint(
                joint=FixtureRobotJoint.ROOT,
                offset=(0.0, 0.0, 0.0),
                axis=(0.0, 1.0, 0.0),
            ),
            FixtureRobotLink.LEFT_FOOT: SimpleKinematicPoint(
                joint=FixtureRobotJoint.LEFT_LEG,
                offset=(-0.15, 0.1, -0.8),
                axis=(0.0, 0.0, 1.0),
            ),
            FixtureRobotLink.RIGHT_FOOT: SimpleKinematicPoint(
                joint=FixtureRobotJoint.RIGHT_LEG,
                offset=(0.15, 0.1, -0.8),
                axis=(0.0, 0.0, 1.0),
            ),
        },
        provenance={"source": "unit_test"},
    )


def fixture_contacts(frame_count: int = 3) -> ContactPlan[
    FixtureContactSubject,
    FixtureContactState,
    FixtureContactPatch,
    FixtureRobotLink,
]:
    return ContactPlan(
        tracks=(
            ContactTrack(
                subject=FixtureContactSubject.LEFT_FOOT,
                patch=FixtureContactPatch.LEFT_SOLE,
                states=tuple(FixtureContactState.SUPPORT for _ in range(frame_count)),
                links=(FixtureRobotLink.LEFT_FOOT,),
                active_states=(FixtureContactState.SUPPORT,),
                support_states=(FixtureContactState.SUPPORT,),
            ),
            ContactTrack(
                subject=FixtureContactSubject.RIGHT_FOOT,
                patch=FixtureContactPatch.RIGHT_SOLE,
                states=tuple(FixtureContactState.AIR for _ in range(frame_count)),
                links=(FixtureRobotLink.RIGHT_FOOT,),
                active_states=(FixtureContactState.SUPPORT,),
                support_states=(FixtureContactState.SUPPORT,),
            ),
        ),
        frame_count=frame_count,
    )


def fixture_targets(frame_count: int = 3) -> LinkTargetPlan[FixtureRobotLink]:
    positions = fixture_motion(frame_count).joint_positions
    return LinkTargetPlan(
        tracks=(
            LinkTargetTrack(
                link=FixtureRobotLink.LEFT_FOOT,
                positions=positions[:, 1],
                weights=np.ones(frame_count),
            ),
            LinkTargetTrack(
                link=FixtureRobotLink.RIGHT_FOOT,
                positions=positions[:, 2],
                weights=np.ones(frame_count),
            ),
        ),
        frame_count=frame_count,
    )


def fixture_problem(frame_count: int = 3) -> RetargetingProblem:
    return RetargetingProblem(
        name="fixture_problem",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=fixture_robot(),
        motion=fixture_motion(frame_count),
        scene=SceneSpec.robot_only(),
        contacts=fixture_contacts(frame_count),
        targets=fixture_targets(frame_count),
        link_bindings=(
            LinkBinding(FixtureMotionJoint.ROOT, FixtureRobotLink.PELVIS),
            LinkBinding(FixtureMotionJoint.LEFT_FOOT, FixtureRobotLink.LEFT_FOOT),
            LinkBinding(FixtureMotionJoint.RIGHT_FOOT, FixtureRobotLink.RIGHT_FOOT),
        ),
        constraints=(
            JointLimitsConstraintConfig(),
            TrustRegionConstraintConfig(),
        ),
        solver=SolverSpec(
            backend=SolverBackend.NUMPY_LEAST_SQUARES,
            max_iterations=2,
        ),
        scale_to_robot=False,
    )
