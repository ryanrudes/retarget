"""Asset manifest management."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
import tomllib
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retarget.core.enums import AssetKind


class AssetRecord(BaseModel):
    """One tracked asset.

    Attributes:
        name (str): Unique asset name in the store manifest.
        kind (AssetKind): Asset category (robot, object, terrain, motion, fixture).
        path (Path): Resolved on-disk path to the asset file or directory.
        source (str | None): Original import path or download URL, if recorded.
        sha256 (str | None): Expected or verified SHA-256 hex digest for file assets.
        license (str | None): SPDX identifier or short license note.
        notice (str | None): Attribution or NOTICE text.
        installed_at (datetime | None): UTC timestamp when the asset was registered or installed.
        metadata (dict[str, Any]): Free-form manifest metadata.
    """

    name: str
    kind: AssetKind
    path: Path
    source: str | None = None
    sha256: str | None = None
    license: str | None = None
    notice: str | None = None
    installed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


class AssetRequirement(BaseModel):
    """One asset declared by a human-editable install manifest.

    Attributes:
        name (str): Asset name to register in the store.
        kind (AssetKind): Asset category (robot, object, terrain, motion, fixture).
        source (str): Local path or HTTP(S) URL to fetch or reference.
        copy_to_store (bool): When ``True`` (TOML key ``copy``), copy into the store instead of
            referencing in place.
        destination (Path | None): Optional store-relative or absolute install path.
        sha256 (str | None): Expected SHA-256 hex digest for file assets.
        license (str | None): SPDX identifier or short license note.
        notice (str | None): Attribution or NOTICE text.
        metadata (dict[str, Any]): Free-form requirement metadata.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    kind: AssetKind
    source: str
    copy_to_store: bool = Field(default=False, alias="copy")
    destination: Path | None = None
    sha256: str | None = None
    license: str | None = None
    notice: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("destination", mode="before")
    @classmethod
    def _coerce_destination(cls, value: Any) -> Path | None:
        return None if value in (None, "") else Path(value)

    @property
    def is_download(self) -> bool:
        """Whether `source` is an HTTP(S) URL."""

        parsed = urlparse(self.source)
        return parsed.scheme in {"http", "https"}

    def resolve_source(self, base_dir: Path) -> AssetRequirement:
        """Resolve local relative sources against a manifest directory."""

        if self.is_download:
            return self
        source_path = Path(self.source)
        if source_path.is_absolute():
            return self
        return self.model_copy(update={"source": str((base_dir / source_path).resolve())})


class AssetInstallManifest(BaseModel):
    """Human-editable manifest describing assets to install or reference.

    Attributes:
        schema_version (int): Manifest format version (default ``1``).
        assets (tuple[AssetRequirement, ...]): Assets to install or reference.
        metadata (dict[str, Any]): Free-form manifest metadata.
    """

    schema_version: int = 1
    assets: tuple[AssetRequirement, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> AssetInstallManifest:
        """Load a TOML, YAML, or JSON install manifest."""

        manifest_path = Path(path)
        data = _load_manifest_mapping(manifest_path)
        manifest = cls.model_validate(data)
        return manifest.resolve_sources(manifest_path.parent)

    def resolve_sources(self, base_dir: Path) -> AssetInstallManifest:
        """Return a copy with local relative sources resolved."""

        return self.model_copy(update={"assets": tuple(asset.resolve_source(base_dir) for asset in self.assets)})


class AssetManifest(BaseModel):
    """Collection of asset records.

    Attributes:
        records (list[AssetRecord]): Installed or referenced assets tracked by the store.
    """

    records: list[AssetRecord] = Field(default_factory=list)

    def find(self, name: str) -> AssetRecord | None:
        """Return the first record by name."""

        return next((record for record in self.records if record.name == name), None)


class AssetStore:
    """Local manifest-backed asset store."""

    def __init__(self, root: str | Path = ".retarget_assets") -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    def load(self) -> AssetManifest:
        """Load manifest, returning an empty manifest if none exists."""

        if not self.manifest_path.exists():
            return AssetManifest()
        data = json.loads(self.manifest_path.read_text())
        return AssetManifest.model_validate(data)

    def save(self, manifest: AssetManifest) -> None:
        """Persist manifest."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(manifest.model_dump_json(indent=2))

    def list(self) -> builtins.list[AssetRecord]:
        """List tracked assets."""

        return self.load().records

    def import_path(
        self,
        source: str | Path,
        *,
        name: str,
        kind: AssetKind,
        copy: bool = False,
        target_path: str | Path | None = None,
        sha256: str | None = None,
        license: str | None = None,
        notice: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssetRecord:
        """Track or copy an asset path into the store."""

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if sha256 is not None:
            _verify_sha256(source_path, sha256)
        target = Path(target_path) if target_path is not None else self.root / kind.value / source_path.name
        target = target if copy else source_path.resolve()
        if copy:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source_path.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source_path, target)
            else:
                shutil.copy2(source_path, target)
        manifest = self.load()
        record = AssetRecord(
            name=name,
            kind=kind,
            path=target,
            source=str(source_path),
            sha256=sha256,
            license=license,
            notice=notice,
            installed_at=_utc_now(),
            metadata=metadata or {},
        )
        manifest.records = [existing for existing in manifest.records if existing.name != name]
        manifest.records.append(record)
        self.save(manifest)
        return record

    def install(self, requirement: AssetRequirement, *, allow_downloads: bool = False) -> AssetRecord:
        """Install or reference one manifest requirement."""

        source = requirement.source
        if requirement.is_download:
            if not allow_downloads:
                raise ValueError(f"Refusing to download {source!r}; pass allow_downloads=True or --allow-downloads")
            source_path = self._download(requirement)
        else:
            source_path = Path(source)
        copy_to_store = requirement.copy_to_store or requirement.is_download or requirement.destination is not None
        target_path = self._resolve_destination(requirement) if copy_to_store else None
        return self.import_path(
            source_path,
            name=requirement.name,
            kind=requirement.kind,
            copy=copy_to_store,
            target_path=target_path,
            sha256=requirement.sha256,
            license=requirement.license,
            notice=requirement.notice,
            metadata=requirement.metadata,
        )

    def install_manifest(
        self,
        manifest: AssetInstallManifest | str | Path,
        *,
        allow_downloads: bool = False,
    ) -> builtins.list[AssetRecord]:
        """Install every asset in a manifest."""

        loaded = AssetInstallManifest.load(manifest) if isinstance(manifest, (str, Path)) else manifest
        return [self.install(requirement, allow_downloads=allow_downloads) for requirement in loaded.assets]

    def _download(self, requirement: AssetRequirement) -> Path:
        parsed = urlparse(requirement.source)
        filename = Path(parsed.path).name or requirement.name
        target = self.root / "downloads" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(requirement.source, target)
        return target

    def _resolve_destination(self, requirement: AssetRequirement) -> Path:
        if requirement.destination is None:
            filename = (
                Path(urlparse(requirement.source).path).name
                if requirement.is_download
                else Path(requirement.source).name
            )
            return self.root / requirement.kind.value / (filename or requirement.name)
        destination = requirement.destination
        return destination if destination.is_absolute() else self.root / destination


def _load_manifest_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return dict(tomllib.loads(path.read_text()))
    if suffix in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping at the document root")
        return dict(loaded)
    if suffix == ".json":
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping at the document root")
        return dict(loaded)
    raise ValueError(f"Unsupported asset manifest suffix {suffix!r}; expected .toml, .yaml, .yml, or .json")


def _verify_sha256(path: Path, expected: str) -> None:
    if path.is_dir():
        raise ValueError("sha256 verification is only supported for files")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")


def _utc_now() -> datetime:
    return datetime.now(UTC)
