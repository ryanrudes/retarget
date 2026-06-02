"""Step 3 — assemble a MotionSequence with foot contacts and export fixture files."""

from __future__ import annotations

from pathlib import Path

from _synthetic import write_fixture_files
from rich.console import Console

DATA_DIR = Path(__file__).resolve().parent / "data"

if __name__ == "__main__":
    write_fixture_files(DATA_DIR)
    Console().print(f"Wrote motion and scene fixtures under [cyan]{DATA_DIR}[/cyan]")
