"""Step 5 — inspect human-to-robot link mapping for a skateboarding problem."""

from __future__ import annotations

from _synthetic import motion_format, synthetic_skate_motion, synthetic_skate_scene
from rich.console import Console

from retarget import RetargetingProblem, Robot, TaskKind
from retarget.robots import robots

if __name__ == "__main__":
    motion = synthetic_skate_motion()
    scene = synthetic_skate_scene()
    problem = RetargetingProblem(
        name="mapping_probe",
        task_kind=TaskKind.OBJECT_INTERACTION,
        robot=robots.get(Robot.G1_LIKE),
        motion=motion,
        motion_format=motion_format(),
        scene=scene,
    )
    Console().print({"resolved_link_mapping": problem.resolved_link_mapping()})
