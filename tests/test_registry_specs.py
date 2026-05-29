import numpy as np
import pytest

from retarget.assets import AssetStore
from retarget.core.enums import AssetKind, RunStatus
from retarget.core.registry import Registry
from retarget.export import ExportResult, ExportSpec, exporters
from retarget.kinematics import kinematics_backends
from retarget.metrics import metrics
from retarget.motion import MotionFormatSpec, MotionSequence, motion_formats, motion_loaders
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
from retarget.results import RetargetingResult
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


def test_protocol_registries_accept_decorated_classes(tmp_path):
    @motion_loaders.register(".unitloader", replace=True)
    class UnitMotionLoader:
        def load(self, path, spec, *, name=None):
            return MotionSequence(
                name=name or path.stem,
                joint_names=spec.joint_names,
                joint_positions=np.zeros((1, len(spec.joint_names), 3), dtype=np.float64),
                fps=spec.default_fps,
                frame=spec.frame_convention,
            )

    @robot_providers.register("unit_test_provider", replace=True)
    class UnitRobotProvider:
        def load(self, name: str, **_kwargs: object) -> RobotSpec:
            return RobotSpec(name=name, dof=1, height_m=1.0, joint_names=("joint",))

    @metrics.register("unit_test_metric", replace=True)
    class UnitMetric:
        name = "unit_test_metric"

        def evaluate(self, _result, _problem=None) -> float:
            return 42.0

    @exporters.register("unit_test_exporter", replace=True)
    class UnitExporter:
        def export(self, result: RetargetingResult, spec: ExportSpec) -> ExportResult:
            spec.output_path.write_text(result.name)
            return ExportResult(
                format_name=spec.format_name,
                path=spec.output_path,
                frame_count=result.frame_count,
                fps=result.fps,
            )

    @visualizers.register("unit_test_visualizer", replace=True)
    class UnitVisualizer:
        def __init__(self) -> None:
            self.names: list[str] = []

        def view(self, result: RetargetingResult) -> None:
            self.names.append(result.name)

    format_spec = motion_formats.get("minimal")
    motion_path = tmp_path / "motion.unitloader"
    motion_path.write_text("")
    result = RetargetingResult(
        name="decorated_result",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((1, 1), dtype=np.float64),
    )

    assert motion_loaders.get(".unitloader").load(motion_path, format_spec).name == "motion"
    assert robot_providers.get("unit_test_provider").load("decorated_robot").name == "decorated_robot"
    assert metrics.get("unit_test_metric").evaluate(result) == 42.0
    assert exporters.get("unit_test_exporter").export(
        result,
        ExportSpec(format_name="unit_test_exporter", output_path=tmp_path / "export.txt"),
    ).path.read_text() == "decorated_result"
    visualizer = visualizers.get("unit_test_visualizer")
    visualizer.view(result)
    assert visualizer.names == ["decorated_result"]
    assert not isinstance(motion_loaders.get(".unitloader"), type)
    assert not isinstance(robot_providers.get("unit_test_provider"), type)
    assert not isinstance(metrics.get("unit_test_metric"), type)
    assert not isinstance(exporters.get("unit_test_exporter"), type)
    assert not isinstance(visualizers.get("unit_test_visualizer"), type)


def test_protocol_registries_validate_bad_extensions():
    with pytest.raises(TypeError, match="MotionLoader"):
        motion_loaders.register(".bad_loader", object(), replace=True)
    with pytest.raises(TypeError, match="RobotProvider"):
        robot_providers.register("bad_provider", object(), replace=True)
    with pytest.raises(TypeError, match="Metric"):
        metrics.register("bad_metric", object(), replace=True)
    with pytest.raises(TypeError, match="Exporter"):
        exporters.register("bad_exporter", object(), replace=True)
    with pytest.raises(TypeError, match="Visualizer"):
        visualizers.register("bad_visualizer", object(), replace=True)


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
