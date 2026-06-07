# Holosoma Climbing

This example runs the Holosoma climbing subset through the same public
`RetargetingExperiment` hierarchy as every other workflow:

```text
MOCAP source
-> HolosomaClimbObservationRecipe
-> SceneObservation
-> HolosomaClimbRetargetingRecipe
-> RetargetingProblem
-> RetargetingResult
```

The experiment requires the Holosoma parity checkout containing
`demo_data/climb/mocap_climb_seq_0` and the G1 spherehand model. By default the
example looks for that checkout at `../holosoma`, beside this repository. The
generic G1 checkout installed by `scripts/bootstrap_robot_assets.py` does not
contain the parity fixture.

Run the Python API example:

```bash
uv run python examples/holosoma/run_retarget.py --frames 120
```

Or run the declarative frontend over the same recipes:

```bash
uv run retarget run --config examples/holosoma/run_config.toml
```

Use `--holosoma-root` in Python or change the three `holosoma_root` values in
the TOML file when the checkout lives elsewhere.
