import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = (
    "basic_robot_only.py",
    "batch_and_evaluate.py",
    "climbing_terrain.py",
    "custom_motion_format.py",
    "custom_objective.py",
    "custom_robot.py",
    "object_interaction.py",
    "skateboarding/run_retarget.py",
)


@pytest.mark.parametrize("example_name", EXAMPLES)
def test_example_script_executes(example_name: str, tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [sys.executable, str(repo_root / "examples" / example_name)],
        check=True,
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
