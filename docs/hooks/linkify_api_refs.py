"""MkDocs hook: linkify inline `code` to mkdocstrings/autorefs targets site-wide."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from docs.hooks.symbol_index import build_autoref_index, resolve_identifier

FENCED_BLOCK_RE = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"(?<!\[)(`)([^`\n]+)\1(?!\])")

SKIP_INLINE_RE = re.compile(
    r"[\s/\\]|\.(?:md|py|yaml|yml|npz|json|toml|txt)\b|^[0-9]+$"
)

_SYMBOL_INDEX: tuple[dict[str, str], dict[str, str]] | None = None


def _symbol_index() -> tuple[dict[str, str], dict[str, str]]:
    global _SYMBOL_INDEX
    if _SYMBOL_INDEX is None:
        _SYMBOL_INDEX = build_autoref_index()
    return _SYMBOL_INDEX


def _linkify_segment(segment: str) -> str:
    by_id, by_short = _symbol_index()

    def repl(match: re.Match[str]) -> str:
        inner = match.group(2)
        if SKIP_INLINE_RE.search(inner):
            return match.group(0)
        target = resolve_identifier(inner, by_id, by_short)
        if not target:
            return match.group(0)
        return f"[`{inner}`][{target}]"

    return INLINE_CODE_RE.sub(repl, segment)


def _linkify_markdown(markdown: str) -> str:
    parts = FENCED_BLOCK_RE.split(markdown)
    for index, part in enumerate(parts):
        if index % 2 == 1:
            continue
        parts[index] = _linkify_segment(part)
    return "".join(parts)


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    if not str(getattr(page.file, "src_path", "")).endswith(".md"):
        return markdown
    return _linkify_markdown(markdown)
