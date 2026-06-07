"""Load a declarative frontend and run the same RetargetingExperiment API."""

from pathlib import Path

from retarget.cli.config import RetargetingRunConfig

repo_root = Path(__file__).resolve().parents[1]
config = RetargetingRunConfig.load(repo_root / "examples" / "run_config.toml")
configured = config.with_overrides(output=Path("declarative_config.npz"))
result = configured.build_experiment().run()
result.save_npz(configured.output)
print(result.qpos.shape)
