import numpy as np
import pytest

from retarget.core.enums import (
    ConvergenceMode,
    NonPenetrationSource,
    ObjectQposMode,
    ObjectSampleSpace,
    SolverBackend,
    TaskKind,
)
from retarget.kinematics import MuJoCoKinematicsBackend
from retarget.optimization import NonPenetrationConstraintConfig
from retarget.pipeline import (
    InteractionMeshRetargetingEngine,
    Retargeter,
    RetargetingExperiment,
)
from retarget.pipeline.compiled import compile_robot
from retarget.recipes.holosoma import (
    G1_LEFT_FOOT_STICKING_LINKS,
    G1_RIGHT_FOOT_STICKING_LINKS,
    HolosomaClimbObservationRecipe,
    HolosomaClimbRetargetingRecipe,
    HolosomaContactSubject,
    default_holosoma_root,
    g1_spherehand_robot,
    holosoma_climb_layout,
    holosoma_g1_robot_dir,
    object_visual_parts_from_urdf,
)

HOLOSOMA_ROOT = default_holosoma_root()
HOLOSOMA_FIXTURE = HOLOSOMA_ROOT / "tests" / "fixtures" / "climb_seq_0"


def test_holosoma_layout_detection_accepts_amazon_repo_shape(tmp_path):
    package_root = tmp_path / "src" / "holosoma_retargeting" / "holosoma_retargeting"
    fixture = package_root / "demo_data" / "climb" / "mocap_climb_seq_0"
    fixture.mkdir(parents=True)
    motion = fixture / "mocap_climb_seq_0_joint_positions_f900-3700.npy"
    motion.write_bytes(b"")
    for name in (
        "multi_boxes.obj",
        "multi_boxes.urdf",
        "g1_29dof_spherehand_w_multi_boxes.xml",
    ):
        (fixture / name).write_text("")
    robot_dir = package_root / "models" / "g1"
    robot_dir.mkdir(parents=True)
    for name in (
        "g1_29dof_spherehand.urdf",
        "g1_29dof_spherehand.xml",
    ):
        (robot_dir / name).write_text("")

    layout = holosoma_climb_layout(tmp_path)

    assert layout.fixture_dir == fixture
    assert layout.motion_path == motion
    assert holosoma_g1_robot_dir(tmp_path) == robot_dir


def test_holosoma_object_visual_parts_parse_urdf_colors(tmp_path):
    (tmp_path / "box_models").mkdir()
    for name in ("box1.obj", "box2.obj"):
        (tmp_path / "box_models" / name).write_text("")
    urdf_path = tmp_path / "multi_boxes.urdf"
    urdf_path.write_text(
        """
<robot name="multi_boxes">
  <link name="box1_link">
    <visual>
      <geometry><mesh filename="box_models/box1.obj" scale="2 3 4"/></geometry>
      <material name="box1_material"><color rgba="0.3 0.7 0.9 0.5"/></material>
    </visual>
  </link>
  <link name="box2_link">
    <visual>
      <geometry><mesh filename="box_models/box2.obj"/></geometry>
      <material name="box2_material"><color rgba="0.7 0.3 0.9 0.5"/></material>
    </visual>
  </link>
</robot>
""".strip()
    )

    parts = object_visual_parts_from_urdf(urdf_path)

    assert tuple(part.name for part in parts) == ("box1", "box2")
    assert parts[0].asset_scale == (2.0, 3.0, 4.0)
    assert parts[0].rgba == pytest.approx((0.3, 0.7, 0.9, 0.5))


@pytest.mark.skipif(
    not HOLOSOMA_FIXTURE.exists(),
    reason="Holosoma climb fixture is unavailable",
)
def test_holosoma_uses_same_observation_and_experiment_hierarchy():
    robot = g1_spherehand_robot(HOLOSOMA_ROOT)
    recipe = HolosomaClimbRetargetingRecipe(
        holosoma_root=HOLOSOMA_ROOT,
        include_object_collision=False,
        solver_backend=SolverBackend.NUMPY_LEAST_SQUARES,
    )
    experiment = RetargetingExperiment(
        observation=HolosomaClimbObservationRecipe.from_fixture(
            HOLOSOMA_ROOT,
            frame_count=3,
        ),
        recipe=recipe,
        robot=robot,
        retargeter_factory=lambda problem: Retargeter(
            engine=InteractionMeshRetargetingEngine(
                kinematics=MuJoCoKinematicsBackend(compile_robot(problem.robot))
            )
        ),
    )

    observation = experiment.observe()
    problem = experiment.build_problem()

    assert observation.timeline.sample_count == 3
    assert observation.contacts is not None
    assert not hasattr(observation.contacts.tracks[0], "link_names")
    assert problem.name == "holosoma_mocap_climb_seq_0"
    assert problem.task_kind == TaskKind.CLIMBING
    assert problem.scene.object is not None
    assert problem.scene.object.qpos_mode == ObjectQposMode.EXTERNAL
    assert problem.scene.object.sample_space == ObjectSampleSpace.OBJECT_ASSET_LOCAL
    assert problem.initial_qpos is not None
    assert problem.initial_qpos.qpos.shape == (3, 36)
    assert problem.solver.max_iterations == 10
    assert problem.solver.first_frame_iterations == 50
    assert problem.solver.convergence == ConvergenceMode.NONE
    nominal = problem.objectives[1]
    assert nominal.qpos_indices == recipe.optimization_policy.nominal_qpos_indices

    result = experiment.run()
    assert result.frame_count == 3
    assert result.qpos.shape == (3, 36)


@pytest.mark.skipif(
    not HOLOSOMA_FIXTURE.exists(),
    reason="Holosoma climb fixture is unavailable",
)
def test_holosoma_resolves_semantic_contacts_and_geometry_policy():
    robot = g1_spherehand_robot(HOLOSOMA_ROOT)
    observation = HolosomaClimbObservationRecipe.from_fixture(
        HOLOSOMA_ROOT,
        frame_count=2,
    ).observe()
    problem = HolosomaClimbRetargetingRecipe(
        holosoma_root=HOLOSOMA_ROOT,
        include_object_collision=True,
    ).build_problem(observation, robot)

    assert observation.contacts is not None
    assert observation.contacts.tracks[0].subject == HolosomaContactSubject.LEFT_FOOT
    assert problem.contacts is not None
    assert problem.contacts.tracks[0].links == G1_LEFT_FOOT_STICKING_LINKS
    assert problem.contacts.tracks[1].links == G1_RIGHT_FOOT_STICKING_LINKS
    non_penetration = problem.constraints[-1]
    assert isinstance(non_penetration, NonPenetrationConstraintConfig)
    assert non_penetration.sources == (NonPenetrationSource.GEOMETRY,)
    assert non_penetration.geometry_pairs
    assert all(pair.first is not pair.second for pair in non_penetration.geometry_pairs)
    assert np.isfinite(problem.motion.joint_positions).all()
