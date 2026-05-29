"""Run batch/evaluation commands programmatically."""

from pathlib import Path

from retarget.cli.main import main

repo_root = Path(__file__).resolve().parents[1]
input_dir = repo_root / "tests" / "fixtures"
output_dir = Path("batch_results")

main(["batch", "--input-dir", str(input_dir), "--pattern", "*.json", "--output-dir", str(output_dir)])
main(
    [
        "evaluate",
        "--batch-manifest",
        str(output_dir / "batch_manifest.json"),
        "--output-dir",
        str(output_dir / "metrics"),
        "--output",
        str(output_dir / "evaluation_manifest.json"),
    ]
)
