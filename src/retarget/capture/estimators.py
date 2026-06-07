"""Replaceable human-pose estimator interfaces."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from retarget.capture.recordings import HumanPoseRecording, VideoRecording
from retarget.capture.sources import GvhmrOutputSource, HumanPoseSourceSchema
from retarget.core.enums import FrameConvention


class HumanPoseEstimator(Protocol):
    """Estimate a typed human-pose recording from video."""

    def estimate(self, recording: VideoRecording) -> HumanPoseRecording:
        """Estimate pose for ``recording``."""


@dataclass(frozen=True)
class VideoPoseSource:
    """Compose a video recording source with a pose estimator."""

    video: VideoRecording
    estimator: HumanPoseEstimator

    def load(self) -> HumanPoseRecording:
        """Estimate and return human pose."""

        return self.estimator.estimate(self.video)


@dataclass(frozen=True)
class GvhmrEstimator:
    """Run a configured local GVHMR checkout in an ephemeral workspace.

    Command entries may contain ``{video}``, ``{workspace}``, ``{output}``, and
    ``{checkout}`` placeholders. The configured command must write
    ``joints.npy`` into ``{output}``.
    """

    checkout: Path
    command: tuple[str, ...]
    schema: HumanPoseSourceSchema
    fps: float
    frame: FrameConvention = FrameConvention.Y_UP_RIGHT_HANDED
    source_height_m: float | None = None

    def estimate(self, recording: VideoRecording) -> HumanPoseRecording:
        """Run GVHMR and load its output before deleting temporary files."""

        checkout = Path(self.checkout).resolve()
        if not checkout.is_dir():
            raise FileNotFoundError(checkout)
        with tempfile.TemporaryDirectory(prefix="retarget-gvhmr-") as temporary:
            workspace = Path(temporary)
            output = workspace / "output"
            output.mkdir()
            video = workspace / recording.path.name
            try:
                video.symlink_to(recording.path.resolve())
            except OSError:
                shutil.copy2(recording.path, video)
            replacements = {
                "video": str(video),
                "workspace": str(workspace),
                "output": str(output),
                "checkout": str(checkout),
            }
            command = tuple(part.format_map(replacements) for part in self.command)
            subprocess.run(command, cwd=checkout, check=True)
            return GvhmrOutputSource(
                path=output,
                schema=self.schema,
                fps=self.fps,
                frame=self.frame,
                name=recording.name,
                source_height_m=self.source_height_m,
            ).load()
