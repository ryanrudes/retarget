"""Imports and schema access for the external motion_sync ecosystem."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]


def load_ecosystem() -> dict[str, Any]:
    """Load motion_sync/contact_detection symbols used by the skateboarding source."""

    for path in (REPO_ROOT / "vendor" / "event_detection" / "src", REPO_ROOT / "vendor" / "motion_sync"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from motion_sync import SyncClip  # type: ignore[import-untyped]
        from motion_sync.schemas.skateboarding import (  # type: ignore[import-untyped]
            SKATE_FOOT_SUPPORT,
            SKATE_SESSION,
            SKATE_VIDEO,
            Bodies,
            SmplxCoreJoints,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Could not import motion_sync/contact_detection. Run git submodule update --init, "
            "or install the sibling packages into this environment."
        ) from exc
    return {
        "SyncClip": SyncClip,
        "SKATE_FOOT_SUPPORT": SKATE_FOOT_SUPPORT,
        "SKATE_SESSION": SKATE_SESSION,
        "SKATE_VIDEO": SKATE_VIDEO,
        "Bodies": Bodies,
        "SmplxCoreJoints": SmplxCoreJoints,
    }
