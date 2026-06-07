"""Robot-resolved link targets for skateboarding observations."""

from __future__ import annotations

import numpy as np

from retarget.motion import LinkTargetPlan, LinkTargetTrack
from retarget.observation import SceneObservation
from retarget.robots.spec import RobotSpec

from .schema import OBSERVATION_TO_ROBOT_ROLE
from .vocabulary import (
    SkateboardingContactState,
    SkateboardingContactSubject,
    SkateboardingObservationRole,
    SkateboardingRigidBody,
)


def skateboarding_link_targets(
    observation: SceneObservation,
    robot: RobotSpec,
) -> LinkTargetPlan:
    """Resolve semantic landmarks and shoe tracks through robot roles."""

    tracks: list[LinkTargetTrack] = []
    for role, weight in (
        (SkateboardingObservationRole.LEFT_HIP, 2.0),
        (SkateboardingObservationRole.RIGHT_HIP, 2.0),
        (SkateboardingObservationRole.LEFT_KNEE, 3.0),
        (SkateboardingObservationRole.RIGHT_KNEE, 3.0),
        (SkateboardingObservationRole.LEFT_ANKLE, 2.0),
        (SkateboardingObservationRole.RIGHT_ANKLE, 2.0),
        (SkateboardingObservationRole.UPPER_BODY_CENTER, 1.0),
    ):
        landmark = observation.landmark(role)
        tracks.append(
            LinkTargetTrack(
                link=robot.link_for_role(OBSERVATION_TO_ROBOT_ROLE[role]),
                positions=landmark.values,
                weights=weight,
                active_mask=landmark.validity,
                provenance={"observation_role": role.value},
            )
        )
    if observation.contacts is None:
        raise ValueError("skateboarding observation requires semantic contacts")
    body_map = {track.role: track for track in observation.rigid_bodies}
    for role, body_role, subject in (
        (
            SkateboardingObservationRole.LEFT_FOOT,
            SkateboardingRigidBody.LEFT_SHOE,
            SkateboardingContactSubject.LEFT_SHOE,
        ),
        (
            SkateboardingObservationRole.RIGHT_FOOT,
            SkateboardingRigidBody.RIGHT_SHOE,
            SkateboardingContactSubject.RIGHT_SHOE,
        ),
    ):
        body = body_map[body_role]
        contact = next(track for track in observation.contacts.tracks if track.subject == subject)
        active = np.asarray(
            [state != SkateboardingContactState.AIR for state in contact.states],
            dtype=bool,
        )
        tracks.append(
            LinkTargetTrack(
                link=robot.link_for_role(OBSERVATION_TO_ROBOT_ROLE[role]),
                positions=body.positions,
                weights=np.where(active, 80.0, 8.0),
                active_mask=body.validity,
                provenance={"observation_role": role.value},
            )
        )
    return LinkTargetPlan(
        tracks=tuple(tracks),
        frame_count=observation.timeline.sample_count,
        provenance={"source": "skateboarding_observation"},
    )
