"""Target-independent Holosoma climbing observation recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from retarget.capture import (
    HumanPoseRecording,
    JointTrack,
    MocapArraySource,
    MocapRecording,
    ObservationSource,
)
from retarget.capture.tracks import PoseTrack
from retarget.core.enums import FrameConvention, ObjectSampleSpace, QuaternionOrder
from retarget.observation import ObservedObject, SceneObservation
from retarget.scene import ObjectSpec

from .contacts import foot_sticking_contacts
from .layout import default_holosoma_root, holosoma_climb_layout
from .motion import preprocess_mocap_climb
from .object import dummy_object_poses, object_visual_parts_from_urdf, sample_multi_boxes_like_holosoma
from .vocabulary import (
    HolosomaGeometryName,
    HolosomaMocapJoint,
    HolosomaObservationRole,
)


@dataclass(frozen=True)
class HolosomaClimbObservationPolicy:
    """Capture-processing choices for the Holosoma climbing observation."""

    source_height_m: float = 1.78
    mat_height_m: float = 0.1
    contact_velocity_threshold: float = 0.01
    object_sample_count: int = 100
    object_sample_seed: int = 42

    def __post_init__(self) -> None:
        if self.source_height_m <= 0.0:
            raise ValueError("source_height_m must be positive")
        if self.mat_height_m < 0.0:
            raise ValueError("mat_height_m must be non-negative")
        if self.contact_velocity_threshold < 0.0:
            raise ValueError("contact_velocity_threshold must be non-negative")
        if self.object_sample_count <= 0:
            raise ValueError("object_sample_count must be positive")


@dataclass(frozen=True)
class HolosomaClimbObservationRecipe:
    """Build a climbing observation from one native mocap source."""

    mocap: ObservationSource[MocapRecording]
    object_mesh_path: Path
    object_urdf_path: Path
    frame_count: int | None = None
    policy: HolosomaClimbObservationPolicy = field(default_factory=HolosomaClimbObservationPolicy)

    @classmethod
    def from_fixture(
        cls,
        holosoma_root: str | Path | None = None,
        *,
        frame_count: int | None = None,
        source_fps: float = 30.0,
        downsample: int = 4,
        policy: HolosomaClimbObservationPolicy | None = None,
    ) -> HolosomaClimbObservationRecipe:
        """Construct the observation recipe for the public Holosoma fixture."""

        if source_fps <= 0.0:
            raise ValueError("source_fps must be positive")
        if downsample <= 0:
            raise ValueError("downsample must be positive")
        root = Path(holosoma_root) if holosoma_root is not None else default_holosoma_root()
        layout = holosoma_climb_layout(root.resolve())
        return cls(
            mocap=MocapArraySource(
                path=layout.motion_path,
                joint_vocabulary=HolosomaMocapJoint,
                fps=source_fps * downsample,
                downsample=downsample,
                name="holosoma_mocap_climb_seq_0",
            ),
            object_mesh_path=layout.object_mesh_path,
            object_urdf_path=layout.object_urdf_path,
            frame_count=frame_count,
            policy=policy or HolosomaClimbObservationPolicy(),
        )

    def observe(self) -> SceneObservation:
        """Load and normalize the native climb capture."""

        mocap = self.mocap.load()
        frame_count = mocap.timeline.sample_count
        if self.frame_count is not None:
            if self.frame_count <= 0:
                raise ValueError("frame_count must be positive")
            frame_count = min(frame_count, self.frame_count)
        timeline = mocap.timeline.slice(slice(0, frame_count))
        joint_names = tuple(track.role.value for track in mocap.joints)
        joint_positions = np.stack([track.values[:frame_count] for track in mocap.joints], axis=1)
        joint_positions = preprocess_mocap_climb(
            joint_positions,
            scale=1.0,
            mat_height=self.policy.mat_height_m,
            demo_joints=joint_names,
        )
        actor = HumanPoseRecording(
            name=mocap.name,
            timeline=timeline,
            frame=FrameConvention.Z_UP_RIGHT_HANDED,
            joints=tuple(
                JointTrack(
                    role=track.role,
                    values=joint_positions[:, index],
                    validity=np.asarray(track.validity, dtype=bool)[:frame_count],
                    provenance=track.provenance,
                )
                for index, track in enumerate(mocap.joints)
            ),
            source_height_m=self.policy.source_height_m,
            provenance=dict(mocap.provenance or {}),
        )
        object_poses = dummy_object_poses(frame_count)
        object_samples = sample_multi_boxes_like_holosoma(
            self.object_mesh_path,
            sample_count=self.policy.object_sample_count,
            seed=self.policy.object_sample_seed,
        )
        observed_object = ObservedObject(
            role=HolosomaObservationRole.CLIMBING_STRUCTURE,
            pose=PoseTrack(
                role=HolosomaObservationRole.CLIMBING_STRUCTURE,
                positions=object_poses[:, 4:],
                quaternions=object_poses[:, :4],
                quaternion_order=QuaternionOrder.WXYZ,
            ),
            geometry=ObjectSpec(
                name=HolosomaGeometryName.MULTI_BOXES.value,
                mesh_path=self.object_mesh_path,
                urdf_path=self.object_urdf_path,
                visual_parts=object_visual_parts_from_urdf(self.object_urdf_path),
                sample_points=object_samples,
                sample_space=ObjectSampleSpace.OBJECT_ASSET_LOCAL,
            ),
            provenance={"source": "holosoma_fixture"},
        )
        contacts = foot_sticking_contacts(
            timeline,
            joint_positions,
            demo_joints=joint_names,
            velocity_threshold=self.policy.contact_velocity_threshold,
        )
        return SceneObservation(
            name=mocap.name,
            timeline=timeline,
            world_frame=FrameConvention.Z_UP_RIGHT_HANDED,
            actor=actor,
            objects=(observed_object,),
            contacts=contacts,
            metadata={"source": "holosoma_fixture"},
        )
