from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from retarget.capture import (
    AlignmentError,
    CategoricalTrack,
    ClockTransform,
    GvhmrEstimator,
    GvhmrOutputSource,
    HumanPoseRecording,
    HumanPoseSourceSchema,
    JointTrack,
    MocapRecording,
    PointTrack,
    PoseTrack,
    SampleTimeline,
    TemporalRegistrationConfig,
    ViconBagSource,
    ViconRecordingSource,
    ViconSourceSchema,
    VideoPoseSource,
    VideoRecording,
    estimate_clock_offset,
    register_rigid_points,
)
from retarget.core.enums import (
    ContactState,
    ContactSubject,
    CropPolicy,
    FrameConvention,
    MocapMarker,
    MocapRigidBody,
    MotionJoint,
    NameEnum,
    Robot,
    SolverBackend,
    TimelineSelection,
)
from retarget.observation import SceneObservation, SemanticContactTrack
from retarget.pipeline import RetargetingExperiment
from retarget.recipes.skateboarding import (
    GVHMR_SCHEMA,
    VICON_SCHEMA,
    SkateboardingContactState,
    SkateboardingMotionJoint,
    SkateboardingObservationRecipe,
    SkateboardingObservationRole,
    SkateboardingRetargetingRecipe,
)
from retarget.recipes.skateboarding.alignment import select_observation_timeline
from retarget.robots import robots


class Joint(MotionJoint):
    ROOT = "root"
    HAND = "hand"


class OtherJoint(MotionJoint):
    ROOT = "other_root"


class Role(NameEnum):
    BODY = "body"


class State(ContactState):
    OFF = "off"
    ON = "on"


class OtherState(ContactState):
    UNKNOWN = "unknown"


class Subject(ContactSubject):
    FOOT = "foot"


class Body(MocapRigidBody):
    SHOE = "shoe"


class Marker(MocapMarker):
    TOE = "toe"


def test_clock_transform_supports_explicit_affine_native_clock_mapping():
    native = SampleTimeline(np.asarray([0.0, 1.0, 2.0]), clock="native")
    transform = ClockTransform(
        scale=1.002,
        offset_s=-0.15,
        source_clock="native",
        target_clock="observation",
    )

    mapped = transform.timeline(native)

    assert mapped.clock == "observation"
    assert np.allclose(mapped.timestamps, [-0.15, 0.852, 1.854])


def test_affine_clock_offset_recovers_known_irregular_offset():
    reference_times = np.sort(np.r_[np.linspace(0.0, 6.0, 121), [0.017, 1.333, 4.444]])
    moving_times = np.linspace(0.0, 6.0, 101)
    offset = 0.37
    reference = np.column_stack([np.sin(reference_times * 2.3), np.cos(reference_times * 0.7)])
    moving = np.column_stack(
        [
            np.sin((moving_times + offset) * 2.3),
            np.cos((moving_times + offset) * 0.7),
        ]
    )

    report = estimate_clock_offset(
        SampleTimeline(reference_times, clock="reference"),
        reference,
        SampleTimeline(moving_times, clock="moving"),
        moving,
        TemporalRegistrationConfig(
            max_abs_offset_s=1.0,
            min_overlap_s=2.0,
            minimum_score=0.8,
        ),
    )

    assert report.accepted
    assert report.clock_transform is not None
    assert report.clock_transform.offset_s == pytest.approx(offset, abs=0.015)


def test_alignment_quality_failure_carries_report():
    times = np.linspace(0.0, 2.0, 50)
    rng = np.random.default_rng(4)

    with pytest.raises(AlignmentError) as exc_info:
        estimate_clock_offset(
            SampleTimeline(times, clock="reference"),
            rng.normal(size=(50, 2)),
            SampleTimeline(times, clock="moving"),
            rng.normal(size=(50, 2)),
            TemporalRegistrationConfig(
                max_abs_offset_s=0.25,
                min_overlap_s=1.0,
                minimum_score=0.95,
            ),
        )

    assert not exc_info.value.report.accepted
    assert exc_info.value.report.clock_transform is not None


def test_track_types_own_their_resampling_policy():
    source = SampleTimeline(np.asarray([0.0, 1.0]), clock="native")
    target = SampleTimeline(np.asarray([0.0, 0.5, 1.0]), clock="native")
    points = PointTrack(
        role=Role.BODY,
        values=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    poses = PoseTrack(
        role=Role.BODY,
        positions=points.values,
        quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    categories = CategoricalTrack(
        role=Role.BODY,
        values=(State.OFF, State.ON),
    )

    point_result = points.resample(source, target)
    pose_result = poses.resample(source, target)
    category_result = categories.resample(source, target)

    assert point_result.values[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert np.linalg.norm(pose_result.quaternions[1]) == pytest.approx(1.0)
    assert category_result.values == (State.OFF, State.OFF, State.ON)


def test_recordings_and_contacts_enforce_enum_families():
    timeline = SampleTimeline.uniform(2, 30.0)
    with pytest.raises(TypeError, match="MotionJoint"):
        HumanPoseRecording(
            name="wrong_vocabulary",
            timeline=timeline,
            frame=FrameConvention.Z_UP_RIGHT_HANDED,
            joints=(
                JointTrack(
                    role=Role.BODY,
                    values=np.zeros((2, 3)),
                ),
            ),
        )

    with pytest.raises(TypeError, match="one ContactState vocabulary"):
        SemanticContactTrack(
            subject=Subject.FOOT,
            states=(State.OFF, OtherState.UNKNOWN),
        )

    with pytest.raises(TypeError, match="one enum vocabulary"):
        HumanPoseRecording(
            name="mixed_vocabulary",
            timeline=timeline,
            frame=FrameConvention.Z_UP_RIGHT_HANDED,
            joints=(
                JointTrack(role=Joint.ROOT, values=np.zeros((2, 3))),
                JointTrack(role=OtherJoint.ROOT, values=np.zeros((2, 3))),
            ),
        )


@pytest.mark.parametrize(
    ("selection", "uniform_fps", "expected"),
    [
        (TimelineSelection.HUMAN_POSE, None, [0.25, 0.5, 0.75]),
        (TimelineSelection.MOCAP, None, [0.5]),
        (TimelineSelection.UNIFORM, 4.0, [0.25, 0.5, 0.75]),
    ],
)
def test_observation_timeline_selection_policies(
    selection,
    uniform_fps,
    expected,
):
    mocap = MocapRecording(
        name="mocap",
        timeline=SampleTimeline(np.asarray([0.0, 0.5, 1.0]), clock="mocap"),
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
    )
    human = HumanPoseRecording(
        name="human",
        timeline=SampleTimeline(np.asarray([0.25, 0.5, 0.75]), clock="human"),
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
        joints=(JointTrack(role=Joint.ROOT, values=np.zeros((3, 3))),),
    )

    timeline = select_observation_timeline(
        mocap,
        human,
        ClockTransform(source_clock="mocap", target_clock="observation"),
        selection=selection,
        crop_policy=CropPolicy.OVERLAP,
        uniform_fps=uniform_fps,
        max_frames=None,
    )

    assert timeline.clock == "observation"
    assert np.allclose(timeline.timestamps, expected)


def test_uniform_timeline_requires_explicit_rate():
    mocap = MocapRecording(
        name="mocap",
        timeline=SampleTimeline.uniform(2, 10.0, clock="mocap"),
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
    )
    human = HumanPoseRecording(
        name="human",
        timeline=SampleTimeline.uniform(2, 10.0, clock="human"),
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
        joints=(JointTrack(role=Joint.ROOT, values=np.zeros((2, 3))),),
    )

    with pytest.raises(ValueError, match="uniform_fps"):
        select_observation_timeline(
            mocap,
            human,
            ClockTransform(source_clock="mocap", target_clock="observation"),
            selection=TimelineSelection.UNIFORM,
            crop_policy=CropPolicy.OVERLAP,
            uniform_fps=None,
            max_frames=None,
        )


def test_rigid_registration_recovers_transform_and_rejects_bad_quality():
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = source @ rotation.T + np.asarray([2.0, 3.0, 4.0])

    transform, report = register_rigid_points(source, target, maximum_rms_error=1e-8)

    assert report.accepted
    assert np.allclose(transform.apply(source), target)

    with pytest.raises(AlignmentError):
        register_rigid_points(
            source,
            target + np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            maximum_rms_error=1e-4,
        )


def _observation() -> SceneObservation:
    timeline = SampleTimeline.uniform(3, 30.0, clock="observation")
    actor = HumanPoseRecording(
        name="checkpoint",
        timeline=timeline,
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
        joints=(
            JointTrack(role=Joint.ROOT, values=np.zeros((3, 3))),
            JointTrack(role=Joint.HAND, values=np.ones((3, 3))),
        ),
        root_pose=PoseTrack(
            role=Joint.ROOT,
            positions=np.zeros((3, 3)),
            quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3),
        ),
    )
    return SceneObservation(
        name="checkpoint",
        timeline=timeline,
        world_frame=FrameConvention.Z_UP_RIGHT_HANDED,
        actor=actor,
    )


def test_scene_observation_checkpoint_is_explicit_and_round_trips(tmp_path):
    observation = _observation()
    path = tmp_path / "observation.npz"

    assert not path.exists()
    observation.save_npz(path)
    restored = SceneObservation.load_npz(path)

    assert path.exists()
    assert restored.timeline == observation.timeline
    assert restored.actor.joint_roles == (Joint.ROOT, Joint.HAND)
    assert restored.actor.root_pose is not None
    assert np.allclose(
        restored.actor.root_pose.quaternions,
        observation.actor.root_pose.quaternions,
    )


def test_video_pose_source_composes_replaceable_estimator(tmp_path):
    video_path = tmp_path / "capture.mp4"
    video_path.write_bytes(b"video")
    video = VideoRecording(
        name="capture",
        path=video_path,
        timeline=SampleTimeline.uniform(2, 30.0, clock="video"),
    )
    expected = HumanPoseRecording(
        name="capture",
        timeline=video.timeline,
        frame=FrameConvention.Y_UP_RIGHT_HANDED,
        joints=(JointTrack(role=Joint.ROOT, values=np.zeros((2, 3))),),
    )

    @dataclass(frozen=True)
    class FakeEstimator:
        result: HumanPoseRecording

        def estimate(self, recording: VideoRecording) -> HumanPoseRecording:
            assert recording is video
            return self.result

    assert VideoPoseSource(video, FakeEstimator(expected)).load() is expected


def test_gvhmr_estimator_uses_ephemeral_workspace(tmp_path):
    checkout = tmp_path / "gvhmr"
    checkout.mkdir()
    video_path = tmp_path / "capture.mp4"
    video_path.write_bytes(b"video")
    video = VideoRecording(
        name="capture",
        path=video_path,
        timeline=SampleTimeline.uniform(2, 30.0, clock="video"),
    )
    script = (
        "from pathlib import Path; import numpy as np; "
        "output=Path(r'{output}'); "
        "np.save(output/'joints.npy', np.zeros((2, 2, 3)))"
    )
    estimator = GvhmrEstimator(
        checkout=checkout,
        command=(sys.executable, "-c", script),
        schema=HumanPoseSourceSchema(joint_indices={Joint.ROOT: 0, Joint.HAND: 1}),
        fps=30.0,
    )

    result = estimator.estimate(video)

    assert result.joint_roles == (Joint.ROOT, Joint.HAND)
    assert np.allclose(result.timeline.timestamps, video.timeline.timestamps)
    assert result.timeline.clock != video.timeline.clock
    assert not Path(str(result.provenance["source_path"])).exists()


def test_vicon_bag_source_deserializes_directly_to_typed_tracks(
    monkeypatch,
    tmp_path,
):
    transform_connection = SimpleNamespace(topic="/tf", msgtype="tf")
    marker_connection = SimpleNamespace(
        topic="/vicon/markers",
        msgtype="markers",
    )
    stamp_one = SimpleNamespace(sec=1, nanosec=0)
    stamp_two = SimpleNamespace(sec=2, nanosec=0)
    transform_message = SimpleNamespace(
        transforms=(
            SimpleNamespace(
                child_frame_id="vicon/shoe/shoe",
                header=SimpleNamespace(stamp=stamp_one),
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                    rotation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
            ),
        )
    )
    marker_message = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp_two),
        markers=(
            SimpleNamespace(
                marker_name="toe",
                occluded=False,
                translation=SimpleNamespace(x=1000.0, y=0.0, z=0.0),
            ),
        ),
    )

    class FakeReader:
        def __init__(self, _paths):
            self.connections = (transform_connection, marker_connection)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def messages(self, *, connections):
            assert connections == [transform_connection, marker_connection]
            yield transform_connection, 1_000_000_000, transform_message
            yield marker_connection, 2_000_000_000, marker_message

        def deserialize(self, rawdata, _msgtype):
            return rawdata

    rosbags = ModuleType("rosbags")
    highlevel = ModuleType("rosbags.highlevel")
    highlevel.AnyReader = FakeReader
    monkeypatch.setitem(sys.modules, "rosbags", rosbags)
    monkeypatch.setitem(sys.modules, "rosbags.highlevel", highlevel)
    bag = tmp_path / "trial"
    bag.mkdir()

    recording = ViconBagSource(
        bag,
        ViconSourceSchema(
            rigid_bodies={"shoe": Body.SHOE},
            markers={"toe": Marker.TOE},
        ),
    ).load()

    assert np.allclose(recording.timeline.timestamps, [0.0, 1.0])
    assert np.allclose(recording.rigid_body(Body.SHOE).positions[0], [1.0, 2.0, 3.0])
    assert recording.rigid_body(Body.SHOE).validity.tolist() == [True, False]
    assert np.allclose(recording.marker(Marker.TOE).values[1], [1.0, 0.0, 0.0])
    assert recording.marker(Marker.TOE).validity.tolist() == [False, True]


REPO_ROOT = Path(__file__).resolve().parents[1]
VICON = REPO_ROOT / "capture_data" / "vicon" / "pushoff5_twoshoes"
GVHMR = REPO_ROOT / "capture_data" / "gvhmr" / "pushoff5_twoshoes"


@pytest.mark.skipif(
    not (VICON / "vicon.npz").exists() or not (GVHMR / "joints.npy").exists(),
    reason="native skateboarding fixture is unavailable",
)
def test_real_skateboarding_capture_observes_without_intermediate_handoff(
    tmp_path,
):
    before = set(tmp_path.iterdir())
    recipe = SkateboardingObservationRecipe(
        mocap=ViconRecordingSource(VICON, VICON_SCHEMA, name="pushoff5_twoshoes"),
        human_pose=GvhmrOutputSource(
            GVHMR,
            GVHMR_SCHEMA,
            fps=59.942,
            name="pushoff5_twoshoes",
        ),
        max_frames=20,
    )

    observation = recipe.observe()

    assert observation.timeline.sample_count == 20
    assert len(observation.alignment_reports) == 2
    assert all(report.accepted for report in observation.alignment_reports)
    assert observation.contacts is not None
    assert observation.alignment_reports[0].score == pytest.approx(
        0.9390183248,
        abs=1e-6,
    )
    assert observation.alignment_reports[1].rms_error == pytest.approx(
        0.0413913612,
        abs=1e-6,
    )
    assert np.allclose(
        observation.observed_object(SkateboardingObservationRole.BOARD).pose.positions[0],
        [-0.74663357, -1.05408470, 0.11170949],
        atol=1e-7,
    )
    assert observation.timeline.timestamps[-1] == pytest.approx(
        0.3169730743,
        abs=1e-9,
    )
    assert np.allclose(
        observation.actor.joint(SkateboardingMotionJoint.PELVIS).values[[0, -1]],
        [
            [-1.80534119, -0.71068665, -0.37344692],
            [-1.80636166, -0.71235210, -0.37245519],
        ],
        atol=1e-7,
    )
    assert all(
        state == SkateboardingContactState.GROUND for track in observation.contacts.tracks for state in track.states
    )

    problem = SkateboardingRetargetingRecipe().build_problem(
        observation,
        robots.get(Robot.G1_LIKE),
    )
    assert problem.targets is not None
    assert problem.targets.link_names == (
        "left_hip_pitch",
        "right_hip_pitch",
        "left_knee",
        "right_knee",
        "left_ankle_pitch",
        "right_ankle_pitch",
        "torso",
        "left_foot",
        "right_foot",
    )
    left_foot_target = problem.targets.tracks[-2]
    assert np.allclose(
        left_foot_target.positions[[0, -1]],
        [
            [-0.9268275773, -0.5787338956, 0.0685823483],
            [-0.9268264286, -0.5787939882, 0.0685285709],
        ],
        atol=1e-8,
    )
    assert np.all(np.asarray(left_foot_target.weights) == 80.0)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.skipif(
    not (VICON / "vicon.npz").exists() or not (GVHMR / "joints.npy").exists(),
    reason="native skateboarding fixture is unavailable",
)
def test_real_skateboarding_capture_runs_end_to_end():
    experiment = RetargetingExperiment(
        observation=SkateboardingObservationRecipe(
            mocap=ViconRecordingSource(
                VICON,
                VICON_SCHEMA,
                name="pushoff5_twoshoes",
            ),
            human_pose=GvhmrOutputSource(
                GVHMR,
                GVHMR_SCHEMA,
                fps=59.942,
                name="pushoff5_twoshoes",
            ),
            max_frames=3,
        ),
        recipe=SkateboardingRetargetingRecipe(
            solver_backend=SolverBackend.NUMPY_LEAST_SQUARES,
        ),
        robot=robots.get(Robot.G1_LIKE),
    )

    result = experiment.run()

    assert result.frame_count == 3
    assert result.qpos.shape[0] == 3
