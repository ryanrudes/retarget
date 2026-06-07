# Assets

The package does not vendor robot, object, or terrain assets. Use an asset store to reference local files or explicitly install copied assets from a manifest.

For the Unitree G1 model used by the skateboarding example, the repository includes a bootstrap script that downloads or reuses Holosoma, copies the G1 URDF/MJCF assets into the ignored asset store, generates a validated `robot.toml`, and registers the asset:

```bash
uv run python scripts/bootstrap_robot_assets.py g1 --store .retarget_assets
```

Use `--holosoma-root /path/to/holosoma` to reuse an existing checkout instead of cloning. After bootstrap, run configs can load the model with `robot = "g1"` and `robot_provider = "asset_store"`.

Import one local path:

```bash
retarget assets import /path/to/g1.xml --name g1 --kind robot --store .retarget_assets
```

Install a manifest:

```bash
retarget assets validate examples/assets_manifest.toml
retarget assets install examples/assets_manifest.toml --store .retarget_assets
```

For real research assets, start from `examples/research_assets_manifest.toml`, replace each `source` with a local absolute path, and fill in the upstream license/NOTICE fields before installing. The template covers G1-style robots, T1-style robots, object-interaction meshes or datasets, and climbing terrain or hold sets without vendoring any of those assets into this repository.

Manifest example:

```toml
[[assets]]
name = "g1"
kind = "robot"
source = "/path/to/g1"
copy = false
license = "See upstream asset license"
notice = "Provide upstream attribution or NOTICE text here."

[[assets]]
name = "climbing_holds"
kind = "terrain"
source = "/path/to/holds"
copy = true
destination = "terrain/climbing_holds"
```

HTTP(S) sources are refused unless `--allow-downloads` is supplied. File `sha256` checks are supported for copied or referenced files. Directory hashes are intentionally not inferred; keep complex dataset integrity checks in the asset provider that knows the dataset layout.

Robot assets can be loaded through
`robot_providers.get(RobotProviderName.ASSET_STORE)` when the registered asset
is a spec file or a directory containing `robot.toml`, `robot.yaml`,
`robot.yml`, or `robot.json`.

Object and terrain assets can be referenced from run specs with `mesh_path`. When explicit `sample_points` are not supplied, the run config or scene spec (`mesh_sample_count` under `[scene.object]` or `[scene.terrain]`) samples deterministic surface points for non-penetration and interaction-mesh constraints. OBJ files are supported dependency-free; other mesh formats need `trimesh`, which is installed with the `mujoco` or `viz` extras (`uv sync --extra mujoco` or `uv sync --extra viz`), not as a standalone extra.

Optional slow tests look for `RETARGET_ASSET_STORE` or `.retarget_assets`. When that store contains `g1`, `g1_like`, `t1`, or `t1_like` robot records, the tests load those specs through the asset-store provider and validate their qpos layout, joint names, and joint limits.
