"""Contact-state inference helpers."""

from __future__ import annotations

import numpy as np

from retarget.motion.spec import MotionFormatSpec, MotionSequence


def infer_contact_by_velocity(
    motion: MotionSequence,
    motion_format: MotionFormatSpec | None,
    *,
    velocity_threshold: float = 0.01,
) -> tuple[dict[str, bool], ...]:
    """Return per-frame binary contact states.

    Explicit contacts stored on the motion sequence take precedence. Otherwise
    contact states are inferred from contact-joint speed; the first and last
    frames reuse their nearest available finite-difference velocity.
    """

    if motion.contacts:
        return _explicit_contacts(motion, motion_format)
    if motion_format is None or not motion_format.contact_joints:
        return tuple({} for _ in range(motion.frame_count))
    if velocity_threshold <= 0:
        raise ValueError("velocity_threshold must be positive")

    contact_names = tuple(name for name in motion_format.contact_joints if name in motion.joint_names)
    if not contact_names:
        return tuple({} for _ in range(motion.frame_count))

    positions = np.stack([motion.joint(name) for name in contact_names], axis=1)
    if motion.frame_count == 1:
        speeds = np.zeros((1, len(contact_names)), dtype=np.float64)
    else:
        velocities = np.gradient(positions, 1.0 / motion.fps, axis=0)
        speeds = np.linalg.norm(velocities, axis=2)
    return tuple(
        {
            name: bool(speeds[frame_idx, contact_idx] <= velocity_threshold)
            for contact_idx, name in enumerate(contact_names)
        }
        for frame_idx in range(motion.frame_count)
    )


def _explicit_contacts(
    motion: MotionSequence,
    motion_format: MotionFormatSpec | None,
) -> tuple[dict[str, bool], ...]:
    if motion_format is None or not motion_format.contact_joints:
        return tuple(dict(frame) for frame in motion.contacts)
    allowed = set(motion_format.contact_joints)
    return tuple(
        {name: bool(active) for name, active in frame.items() if name in allowed}
        for frame in motion.contacts
    )
