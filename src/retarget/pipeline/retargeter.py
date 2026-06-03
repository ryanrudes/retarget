"""High-level retargeting runner."""

from __future__ import annotations

import time

from retarget.pipeline.engine import InteractionMeshRetargetingEngine, result_from_engine_output
from retarget.pipeline.problem import RetargetingProblem
from retarget.results.spec import RetargetingResult


class Retargeter:
    """Run a `RetargetingProblem` and return a typed result."""

    def __init__(self, *, engine: InteractionMeshRetargetingEngine | None = None) -> None:
        self.engine: InteractionMeshRetargetingEngine = engine or InteractionMeshRetargetingEngine()

    def run(self, problem: RetargetingProblem) -> RetargetingResult:
        """Retarget a motion sequence."""

        start = time.perf_counter()
        prepared = problem.with_output_fps_applied()
        output = self.engine.run(prepared)
        return result_from_engine_output(prepared, output, runtime_s=time.perf_counter() - start)
