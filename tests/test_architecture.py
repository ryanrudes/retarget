"""Repository-level guards for the typed domain architecture."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "retarget"


def _python_sources(root: Path = SOURCE) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def test_source_has_no_literal_discriminators_or_pickle_loading():
    violations: list[str] = []
    for path in _python_sources():
        text = path.read_text()
        for forbidden in ("Literal[", "allow_pickle=True", "allow_pickle = True"):
            if forbidden in text:
                violations.append(f"{path.relative_to(ROOT)}: {forbidden}")
    assert not violations, "\n".join(violations)


def test_public_domain_models_do_not_reintroduce_string_semantic_fields():
    domain_paths = (
        SOURCE / "motion" / "spec.py",
        SOURCE / "motion" / "contact.py",
        SOURCE / "motion" / "targets.py",
        SOURCE / "observation" / "spec.py",
        SOURCE / "observation" / "contact.py",
        SOURCE / "pipeline" / "problem.py",
        SOURCE / "robots" / "spec.py",
    )
    forbidden_fields = (
        "joint_names",
        "link_names",
        "contact_links: tuple[str",
        "nominal_tracking_joints",
        "metadata:",
    )
    violations = [
        f"{path.relative_to(ROOT)}: {field}"
        for path in domain_paths
        for field in forbidden_fields
        if field in path.read_text()
    ]
    assert not violations, "\n".join(violations)


def test_runtime_has_no_semantic_name_heuristics_or_implicit_geometry_pairs():
    runtime_paths = (
        SOURCE / "kinematics" / "backends.py",
        SOURCE / "pipeline" / "compiled.py",
        SOURCE / "pipeline" / "problem.py",
        SOURCE / "optimization" / "terms.py",
    )
    heuristic_patterns = (
        r"\.lower\(\)",
        r"\.casefold\(\)",
        r"\.startswith\(",
        r"\.endswith\(",
        r"_default_geom_pairs",
        r"G1_BODY_ALIASES",
        r"strip_floor_contact_pairs",
    )
    violations: list[str] = []
    for path in runtime_paths:
        text = path.read_text()
        for pattern in heuristic_patterns:
            if re.search(pattern, text):
                violations.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert not violations, "\n".join(violations)


def test_removed_adapter_and_synchronization_apis_do_not_return():
    forbidden = (
        "SyncClip",
        "PreparedRetargetingInputs",
        "RetargetingSource",
        "from_sync_clip",
        "from_skateboarding_clip",
        "motion_sync",
        "contact_detection",
    )
    violations = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in _python_sources()
        for name in forbidden
        if name in path.read_text()
    ]
    assert not violations, "\n".join(violations)
