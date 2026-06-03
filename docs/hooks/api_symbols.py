"""MkDocs hook: index mkdocstrings API anchors for hover/click navigation in code blocks."""

from __future__ import annotations

import html
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("mkdocs.plugins.api_symbols")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from docs.hooks.symbol_index import LOWERCASE_SHORT_NAMES, SECTION_SUFFIXES, write_anchor_cache

_API_ID_PREFIX = r"(?:retarget|motion_sync|contact_detection)"
ANCHOR_RE = re.compile(rf'\bid="({_API_ID_PREFIX}[^"]+)"')
AUTOREF_TITLE_RE = re.compile(
    r'<a class="autorefs[^"]*" title="([^"]*)" href="[^"]*#([^"]+)"'
)
HEADING_TITLE_RE = re.compile(
    rf'<h[1-6] id="({_API_ID_PREFIX}[^"]+)" class="doc doc-heading"[^>]*>(.*?)</h[1-6]>',
    re.DOTALL,
)


def _page_priority(page_url: str) -> int:
    if page_url.endswith("/api/reference/"):
        return 100
    if page_url.endswith("/api/motion-sync/"):
        return 45
    if page_url.endswith("/api/motion-sync-reference/"):
        return 48
    if page_url.endswith("/api/contact-detection/"):
        return 45
    if page_url.endswith("/api/models/"):
        return 80
    if page_url.endswith("/api/pipeline/"):
        return 75
    if page_url.endswith("/api/enums/"):
        return 55
    if page_url.endswith("/api/asset-store/"):
        return 55
    if page_url.endswith("/api/export-api/"):
        return 55
    if page_url.endswith("/api/pose/"):
        return 55
    if page_url.endswith("/api/"):
        return 90
    return 50


def _is_primary_symbol(anchor_id: str) -> bool:
    if anchor_id in ("retarget", "motion_sync", "contact_detection"):
        return True
    if any(anchor_id.endswith(suffix) for suffix in SECTION_SUFFIXES):
        return False
    return (
        anchor_id.startswith("retarget.")
        or anchor_id.startswith("motion_sync.")
        or anchor_id.startswith("contact_detection.")
    )


def _short_name(anchor_id: str) -> str | None:
    short = anchor_id.rsplit(".", 1)[-1]
    if short[0].isupper():
        return short
    if short in LOWERCASE_SHORT_NAMES:
        return short
    if "_" in short and short.islower():
        return short
    return None


def _normalize_heading_tooltip(inner: str) -> str:
    inner = re.sub(
        r'<a href="#[^"]*" class="headerlink"[^>]*>.*?</a>',
        "",
        inner,
        flags=re.DOTALL,
    )
    inner = re.sub(r"\s+", " ", inner).strip()
    return inner


def _scrape_tooltip_titles(site_dir: Path) -> dict[str, str]:
    """Rich autoref tooltip HTML (symbol kind badge + name), keyed by anchor id."""
    titles: dict[str, str] = {}

    for html_path in site_dir.glob("**/*.html"):
        text = html_path.read_text(encoding="utf-8")

        for match in AUTOREF_TITLE_RE.finditer(text):
            titles[match.group(2)] = html.unescape(match.group(1))

        if "/api/" not in html_path.as_posix():
            continue

        for match in HEADING_TITLE_RE.finditer(text):
            anchor_id = match.group(1)
            if anchor_id in titles:
                continue
            inner = _normalize_heading_tooltip(match.group(2))
            if "doc-symbol" in inner:
                titles[anchor_id] = inner

    return titles


def _scan_api_pages(
    site_dir: Path,
) -> tuple[dict[str, str], dict[str, str], set[str], dict[str, str]]:
    by_id: dict[str, tuple[str, int]] = {}
    by_short: dict[str, tuple[str, int, str]] = {}
    anchor_ids: set[str] = set()

    for html_path in sorted(site_dir.glob("api/**/index.html")):
        page_url = "/" + html_path.relative_to(site_dir).parent.as_posix() + "/"
        priority = _page_priority(page_url)
        text = html_path.read_text(encoding="utf-8")

        for match in ANCHOR_RE.finditer(text):
            anchor_id = match.group(1)
            if not _is_primary_symbol(anchor_id):
                continue
            anchor_ids.add(anchor_id)

            href = f"{page_url}#{anchor_id}"
            existing = by_id.get(anchor_id)
            if existing is None or priority < existing[1]:
                by_id[anchor_id] = (href, priority)

            short = _short_name(anchor_id)
            if short is None:
                continue
            existing_short = by_short.get(short)
            if existing_short is None or priority < existing_short[1]:
                by_short[short] = (href, priority, anchor_id)

    titles = _scrape_tooltip_titles(site_dir)

    return (
        {key: href for key, (href, _) in by_id.items()},
        {key: href for key, (href, _, _) in by_short.items()},
        anchor_ids,
        titles,
    )


def _write_symbol_index(config: dict[str, Any]) -> None:
    site_dir = Path(config["site_dir"])
    by_id, by_short, anchor_ids, titles = _scan_api_pages(site_dir)
    write_anchor_cache(anchor_ids)

    titles_by_short: dict[str, str] = {}
    for short, href in by_short.items():
        anchor_id = href.split("#", 1)[-1]
        if anchor_id in titles:
            titles_by_short[short] = titles[anchor_id]

    payload = {
        "version": 2,
        "byId": by_id,
        "byShortName": by_short,
        "titles": titles,
        "titlesByShortName": titles_by_short,
    }
    encoded = json.dumps(payload, separators=(",", ":"))

    for base in (site_dir, Path(config["docs_dir"])):
        out_path = base / "javascripts" / "api-symbols.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")

    log.info(
        "API symbol index: %d anchors, %d tooltip titles",
        len(by_id),
        len(titles),
    )


def on_pre_build(config: dict[str, Any], **kwargs: Any) -> None:
    """Seed serve temp dir from last build so tooltips work before post_build runs."""
    docs_copy = Path(config["docs_dir"]) / "javascripts" / "api-symbols.json"
    if not docs_copy.is_file():
        return
    site_copy = Path(config["site_dir"]) / "javascripts" / "api-symbols.json"
    site_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(docs_copy, site_copy)


def on_post_build(config: dict[str, Any], **kwargs: Any) -> None:
    _write_symbol_index(config)
