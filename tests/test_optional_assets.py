import os
from pathlib import Path

import pytest

from retarget.assets import AssetStore
from retarget.robots import robot_providers


@pytest.mark.assets
@pytest.mark.slow
def test_optional_g1_t1_asset_specs_load_when_asset_store_is_present():
    store_root = Path(os.environ.get("RETARGET_ASSET_STORE", ".retarget_assets"))
    manifest_path = store_root / "manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"no asset store manifest at {manifest_path}")

    store = AssetStore(store_root)
    manifest = store.load()
    candidate_names = ("g1", "g1_like", "t1", "t1_like")
    available_names = [name for name in candidate_names if manifest.find(name) is not None]
    if not available_names:
        pytest.skip("asset store has no G1/T1-style robot assets")

    provider = robot_providers.get("asset_store")
    for name in available_names:
        robot = provider.load(name, store=store_root)
        assert robot.dof > 0
        assert robot.qpos_size() >= robot.dof
        assert robot.joint_names
        assert robot.joint_limits
