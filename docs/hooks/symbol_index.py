"""Build retarget API symbol ids for autorefs / api-navigation (shared by hooks)."""

from __future__ import annotations

import inspect
import json
import re
from collections import defaultdict
from pathlib import Path

import retarget

# Registry singletons re-exported from `retarget`.
LOWERCASE_SHORT_NAMES = frozenset(
    {
        "constraint_terms",
        "exporters",
        "kinematics_backends",
        "metrics",
        "motion_formats",
        "motion_loaders",
        "objective_terms",
        "robot_providers",
        "robots",
        "solver_factories",
        "visualizers",
    }
)

# Top-level `Registry` exports have no stable qualname; map to mkdocstrings anchors.
REGISTRY_EXPORT_ANCHORS: dict[str, str] = {
    "constraint_terms": "retarget.optimization.constraint_terms",
    "exporters": "retarget.export.exporters",
    "kinematics_backends": "retarget.kinematics.kinematics_backends",
    "metrics": "retarget.metrics.metrics",
    "motion_formats": "retarget.motion.motion_formats",
    "motion_loaders": "retarget.motion.motion_loaders",
    "objective_terms": "retarget.optimization.objective_terms",
    "robot_providers": "retarget.robots.robot_providers",
    "robots": "retarget.robots.robots",
    "solver_factories": "retarget.optimization.solver_factories",
    "visualizers": "retarget.visualization.visualizers",
}

AMBIGUOUS_SHORT_NAMES = frozenset(
    {
        "config",
        "data",
        "engine",
        "format",
        "frame",
        "info",
        "input",
        "kind",
        "load",
        "meta",
        "motion",
        "name",
        "object",
        "output",
        "path",
        "problem",
        "record",
        "report",
        "result",
        "run",
        "scale",
        "spec",
        "term",
        "terms",
        "test",
        "type",
        "value",
    }
)

# When several anchors share a short name, prefer these parent types.
PREFER_CLASS_FOR_SHORT: dict[str, str] = {
    "joint_names": "MotionSequence",
    "joint_positions": "MotionSequence",
}

FIELD_SHORT_BLOCKLIST = frozenset(
    {
        "contacts",
        "fps",
        "height_m",
        "metadata",
        "mesh_path",
        "name",
    }
)

SECTION_SUFFIXES = ("-functions", "-attributes", "-classes")
ANCHOR_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "api-anchor-ids.json"


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _qualname(obj: object) -> str | None:
    if inspect.ismodule(obj):
        name = getattr(obj, "__name__", "")
        return name if name.startswith("retarget") else None
    module = getattr(obj, "__module__", None)
    qual = getattr(obj, "__qualname__", None)
    if not module or not qual or not str(module).startswith("retarget"):
        return None
    return f"{module}.{qual}"


def _short_name(anchor_id: str) -> str | None:
    short = anchor_id.rsplit(".", 1)[-1]
    if short in AMBIGUOUS_SHORT_NAMES or short in FIELD_SHORT_BLOCKLIST:
        return None
    if short[0].isupper():
        return short
    if short in LOWERCASE_SHORT_NAMES:
        return short
    if "_" in short and short.islower():
        return short
    return None


def _register_short(
    by_short: dict[str, str],
    rank: dict[str, tuple[int, int]],
    anchor_id: str,
) -> None:
    short = _short_name(anchor_id)
    if short is None:
        return
    depth = len(anchor_id.split("."))
    preferred = PREFER_CLASS_FOR_SHORT.get(short)
    if preferred and f".{preferred}." in anchor_id:
        tier = 0
    elif preferred:
        tier = 1
    else:
        tier = 0
    priority = (tier, depth)
    if short in rank and rank[short] <= priority:
        return
    rank[short] = priority
    by_short[short] = anchor_id


def _export_entries() -> list[tuple[str, bool]]:
    entries: list[tuple[str, bool]] = []
    for name in retarget.__all__:
        obj = getattr(retarget, name)
        full = (
            _qualname(obj)
            or REGISTRY_EXPORT_ANCHORS.get(name)
            or f"retarget.{name}"
        )
        entries.append((full, True))
    for module in (retarget.motion,):
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            full = _qualname(obj)
            if full:
                entries.append((full, True))
    return entries


def build_conservative_index() -> tuple[dict[str, str], dict[str, str]]:
    """First-build fallback: top-level exports + motion helpers only."""
    by_id: dict[str, str] = {}
    by_short: dict[str, str] = {}
    rank: dict[str, tuple[int, int]] = {}
    for full_id, _exported in _export_entries():
        if not full_id.startswith("retarget."):
            continue
        by_id[full_id] = full_id
        _register_short(by_short, rank, full_id)
    return by_id, by_short


def build_index_from_anchor_ids(anchor_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    by_id = {anchor_id: anchor_id for anchor_id in sorted(anchor_ids)}
    by_short: dict[str, str] = {}
    rank: dict[str, tuple[int, int]] = {}
    for anchor_id in anchor_ids:
        _register_short(by_short, rank, anchor_id)

    # Drop ambiguous short names.
    collisions: dict[str, list[str]] = defaultdict(list)
    for short, full in list(by_short.items()):
        collisions[short].append(full)
    for short, ids in collisions.items():
        unique = list(dict.fromkeys(ids))
        if len(unique) == 1:
            continue
        preferred = PREFER_CLASS_FOR_SHORT.get(short)
        if preferred:
            matches = [
                item
                for item in unique
                if f".{preferred}." in item or item.endswith(f".{preferred}")
            ]
            if matches:
                by_short[short] = matches[0]
                continue
        del by_short[short]
    return by_id, by_short


def load_cached_anchor_ids() -> set[str] | None:
    if not ANCHOR_CACHE.is_file():
        return None
    try:
        payload = json.loads(ANCHOR_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    return {item for item in payload if isinstance(item, str) and item.startswith("retarget.")}


def write_anchor_cache(anchor_ids: set[str]) -> None:
    ANCHOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ANCHOR_CACHE.write_text(
        json.dumps(sorted(anchor_ids), indent=0),
        encoding="utf-8",
    )


def build_autoref_index() -> tuple[dict[str, str], dict[str, str]]:
    cached = load_cached_anchor_ids()
    if cached:
        return build_index_from_anchor_ids(cached)
    return build_conservative_index()


def resolve_identifier(text: str, by_id: dict[str, str], by_short: dict[str, str]) -> str | None:
    if text in by_id:
        return text
    if not re.fullmatch(r"[A-Za-z_][\w.]*", text):
        return None
    if text in by_short:
        return by_short[text]
    if "." in text and text.startswith("retarget."):
        return by_id.get(text)
    return None
