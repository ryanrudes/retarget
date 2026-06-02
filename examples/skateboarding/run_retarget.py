"""End-to-end skateboarding retarget (object interaction + contacts + moving board)."""

from __future__ import annotations

from pathlib import Path

from _synthetic import motion_format, synthetic_skate_motion, synthetic_skate_scene
from rich.console import Console

from retarget import (
    Constraint,
    ConstraintSpec,
    OptimizationProfile,
    Retargeter,
    RetargetingProblem,
    Robot,
    SolverBackend,
    SolverSpec,
    TaskKind,
)
from retarget.robots import robots
from retarget.visualization import view_result

OUTPUT = Path("skateboarding_retarget.npz")

if __name__ == "__main__":
    console = Console()
    motion = synthetic_skate_motion()
    scene = synthetic_skate_scene()
    profile = OptimizationProfile(
        name="skateboarding_fixture",
        objectives=OptimizationProfile.defaults().objectives,
        constraints=(
            ConstraintSpec(name=Constraint.JOINT_LIMITS),
            ConstraintSpec(name=Constraint.TRUST_REGION),
        ),
    )
    problem = RetargetingProblem(
        name="synthetic_skate_clip",
        task_kind=TaskKind.OBJECT_INTERACTION,
        robot=robots.get(Robot.SYNTHETIC_HUMANOID),
        motion=motion,
        motion_format=motion_format(),
        scene=scene,
        solver=SolverSpec(backend=SolverBackend.NUMPY_LEAST_SQUARES, max_iterations=4, trust_radius=0.2),
        objectives=profile.objectives,
        constraints=profile.constraints,
        scale_to_robot=True,
        output_fps=motion.fps,
    )
    result = Retargeter().run(problem)
    result.save_npz(OUTPUT)
    console.print(f"Saved [bold]{result.name}[/bold] to [cyan]{OUTPUT}[/cyan]")
    view_result(result, dry_run=True)
    console.print(f"Open the live visualizer with: [bold]uv run retarget view --result {OUTPUT} --live[/bold]")
