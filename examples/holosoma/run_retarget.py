"""Run the Holosoma climbing subset through the unified experiment API."""

from __future__ import annotations

import argparse
from pathlib import Path

from retarget import (
    InteractionMeshRetargetingEngine,
    Retargeter,
    RetargetingExperiment,
)
from retarget.kinematics import MuJoCoKinematicsBackend
from retarget.pipeline.compiled import compile_robot
from retarget.recipes.holosoma import (
    HolosomaClimbObservationRecipe,
    HolosomaClimbRetargetingRecipe,
    default_holosoma_root,
    g1_spherehand_robot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holosoma-root",
        type=Path,
        default=default_holosoma_root(),
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "examples" / "holosoma_climb_120.npz",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.holosoma_root.expanduser().resolve()
    experiment = RetargetingExperiment(
        observation=HolosomaClimbObservationRecipe.from_fixture(
            root,
            frame_count=args.frames,
        ),
        recipe=HolosomaClimbRetargetingRecipe(
            holosoma_root=root,
            show_progress=args.progress,
        ),
        robot=g1_spherehand_robot(root),
        retargeter_factory=lambda problem: Retargeter(
            engine=InteractionMeshRetargetingEngine(
                kinematics=MuJoCoKinematicsBackend(compile_robot(problem.robot))
            )
        ),
    )
    result = experiment.run()
    result.save_npz(args.output)
    print(f"Saved {result.name} to {args.output}")


if __name__ == "__main__":
    main()
