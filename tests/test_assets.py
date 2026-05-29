import hashlib

import pytest

from retarget.assets import AssetInstallManifest, AssetStore
from retarget.core.enums import AssetKind


def test_asset_install_manifest_resolves_and_installs_local_assets(tmp_path):
    source = tmp_path / "source" / "fixture.txt"
    source.parent.mkdir()
    source.write_text("fixture")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "assets.toml"
    manifest_path.write_text(
        f"""
[[assets]]
name = "tiny_fixture"
kind = "fixture"
source = "source/fixture.txt"
copy = true
destination = "fixtures/tiny_fixture.txt"
sha256 = "{expected_hash}"
license = "Apache-2.0"
notice = "Synthetic fixture created for tests."
""".strip()
    )

    manifest = AssetInstallManifest.load(manifest_path)
    records = AssetStore(tmp_path / "store").install_manifest(manifest)

    assert manifest.assets[0].source == str(source.resolve())
    assert len(records) == 1
    assert records[0].kind == AssetKind.FIXTURE
    assert records[0].path == tmp_path / "store" / "fixtures" / "tiny_fixture.txt"
    assert records[0].path.read_text() == "fixture"
    assert records[0].sha256 == expected_hash
    assert AssetStore(tmp_path / "store").load().find("tiny_fixture") is not None


def test_asset_install_refuses_downloads_without_opt_in(tmp_path):
    manifest_path = tmp_path / "assets.toml"
    manifest_path.write_text(
        """
[[assets]]
name = "remote"
kind = "object"
source = "https://example.invalid/object.zip"
""".strip()
    )

    with pytest.raises(ValueError, match="Refusing to download"):
        AssetStore(tmp_path / "store").install_manifest(manifest_path)


def test_research_asset_manifest_template_is_valid():
    manifest = AssetInstallManifest.load("examples/research_assets_manifest.toml")

    assert {asset.name for asset in manifest.assets} == {
        "g1",
        "t1",
        "interaction_object_set",
        "climbing_hold_set",
    }
    assert all(asset.license for asset in manifest.assets)
    assert all(asset.notice for asset in manifest.assets)
