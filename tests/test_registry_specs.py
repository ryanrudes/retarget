import numpy as np
import pytest

from retarget.assets import AssetStore
from retarget.core.enums import AssetKind
from retarget.core.registry import Registry
from retarget.kinematics import kinematics_backends
from retarget.metrics import metrics
from retarget.motion import MotionFormatSpec, motion_formats, motion_loaders
from retarget.optimization import (
    ConstraintContribution,
    ConstraintSpec,
    ObjectiveContribution,
    ObjectiveSpec,
    QuadraticProblem,
    SolverResult,
    SolverSpec,
    TermContext,
    constraint_terms,
    create_solver,
    objective_terms,
    solver_factories,
)
from retarget.robots import RobotSpec, robot_providers, robots
from retarget.visualization import visualizers


def test_registry_decorator():
    registry: Registry[int] = Registry("numbers")

    @registry.register("one")
    def value() -> int:
        return 1

    assert registry.get("one")() == 1


def test_registry_duplicate_rejected():
    registry: Registry[int] = Registry("numbers")
    registry.register("one", 1)
    with pytest.raises(KeyError):
        registry.register("one", 1)


def test_registry_missing_preserves_first_seen_order():
    registry: Registry[int] = Registry("numbers")
    registry.register("one", 1)

    assert registry.missing(("two", "one", "two", "three")) == ("two", "three")
    with pytest.raises(KeyError, match="Available: one"):
        registry.require_all(("two",))


def test_builtin_specs_available():
    assert motion_formats.get("minimal").root_joint == "Pelvis"
    assert ".csv" in motion_loaders.names()
    assert robots.get("synthetic_humanoid").dof > 0
    g1 = robots.get("g1_like")
    t1 = robots.get("t1_like")
    assert g1.dof == 29
    assert t1.dof == 23
    assert "left_ankle_pitch" in g1.joint_names
    assert "torso_yaw" in t1.joint_names
    assert g1.default_link_mapping["L_Toe"] == "left_foot"
    assert "placeholder" not in g1.metadata["description"]


def test_spec_registries_accept_decorated_factories():
    @motion_formats.register("unit_test_format", replace=True)
    def unit_test_format() -> MotionFormatSpec:
        return MotionFormatSpec(
            name="unit_test_format",
            joint_names=("root", "left_toe", "right_toe"),
            root_joint="root",
            contact_joints=("left_toe", "right_toe"),
        )

    @robots.register("unit_test_robot", replace=True)
    def unit_test_robot() -> RobotSpec:
        return RobotSpec(
            name="unit_test_robot",
            dof=1,
            height_m=1.0,
            joint_names=("joint",),
            link_names=("root",),
            joint_limits={"joint": (-1.0, 1.0)},
        )

    assert motion_formats.get("unit_test_format").root_joint == "root"
    assert robots.get("unit_test_robot").joint_names == ("joint",)


def test_builtin_extension_registries_available():
    assert "laplacian" in objective_terms.names()
    assert "joint_limits" in constraint_terms.names()
    assert "numpy_least_squares" in solver_factories.names()
    assert "simple" in kinematics_backends.names()
    assert "optimization_cost" in metrics.names()
    assert "dry_run" in visualizers.names()
    assert robot_providers.get("registry").load("synthetic_humanoid").name == "synthetic_humanoid"
    assert "file" in robot_providers.names()
    assert "asset_store" in robot_providers.names()


def test_objective_and_constraint_registries_accept_decorated_classes():
    @objective_terms.register("unit_test_decorated_objective", replace=True)
    class DecoratedObjective:
        name = "unit_test_decorated_objective"

        def describe(self) -> str:
            return "Decorated objective."

        def build(self, _context: TermContext, _spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]:
            return ()

    @constraint_terms.register("unit_test_decorated_constraint", replace=True)
    class DecoratedConstraint:
        name = "unit_test_decorated_constraint"

        def describe(self) -> str:
            return "Decorated constraint."

        def build(self, _context: TermContext, _spec: ConstraintSpec) -> ConstraintContribution:
            return ConstraintContribution()

    assert objective_terms.get("unit_test_decorated_objective").describe() == "Decorated objective."
    assert constraint_terms.get("unit_test_decorated_constraint").describe() == "Decorated constraint."
    assert not isinstance(objective_terms.get("unit_test_decorated_objective"), type)
    assert not isinstance(constraint_terms.get("unit_test_decorated_constraint"), type)


def test_objective_and_constraint_registries_validate_protocols():
    with pytest.raises(TypeError, match="ObjectiveTerm"):
        objective_terms.register("unit_test_bad_objective", object(), replace=True)
    with pytest.raises(TypeError, match="ConstraintTerm"):
        constraint_terms.register("unit_test_bad_constraint", object(), replace=True)


def test_robot_spec_file_provider_resolves_relative_asset_paths(tmp_path):
    spec_path = tmp_path / "robot.toml"
    spec_path.write_text(
        """
name = "file_bot"
dof = 1
height_m = 1.0
joint_names = ["joint"]
link_names = ["link"]
contact_links = ["link"]
mujoco_xml_path = "robot.xml"

[joint_limits]
joint = [-1.0, 1.0]
""".strip()
    )
    (tmp_path / "robot.xml").write_text("<mujoco/>")

    robot = robot_providers.get("file").load("file_bot", path=spec_path)

    assert robot.name == "file_bot"
    assert robot.mujoco_xml_path == tmp_path / "robot.xml"


def test_robot_asset_store_provider_loads_robot_spec_from_asset_directory(tmp_path):
    asset_dir = tmp_path / "asset_robot"
    asset_dir.mkdir()
    (asset_dir / "robot.toml").write_text(
        """
name = "asset_bot"
dof = 1
height_m = 1.0
joint_names = ["joint"]
link_names = ["link"]

[joint_limits]
joint = [-1.0, 1.0]
""".strip()
    )
    store = AssetStore(tmp_path / "store")
    store.import_path(asset_dir, name="asset_bot", kind=AssetKind.ROBOT)

    robot = robot_providers.get("asset_store").load("asset_bot", store=tmp_path / "store")

    assert robot.name == "asset_bot"
    assert robot.joint_names == ("joint",)


def test_custom_solver_backend_registration():
    class ZeroSolver:
        def solve(self, problem: QuadraticProblem) -> SolverResult:
            return SolverResult.from_solution(
                problem,
                np.zeros(problem.matrix.shape[1], dtype=np.float64),
                status="unit_test_zero",
            )

    @solver_factories.register("unit_test_zero_solver", replace=True)
    def make_zero_solver(_spec: SolverSpec) -> ZeroSolver:
        return ZeroSolver()

    solver = create_solver(SolverSpec(backend="unit_test_zero_solver"))
    result = solver.solve(QuadraticProblem(matrix=np.eye(2), target=np.ones(2)))
    assert result.status == "unit_test_zero"
    assert np.array_equal(result.solution, np.zeros(2))
