"""MkDocs hook: TOC entries use decorator labels (property, classmethod) when present on headings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_HEADING_RE = re.compile(
    r'<h([1-6])\s+id="([^"]+)"\s+class="doc doc-heading"[^>]*>(.*?)</h\1>',
    re.DOTALL,
)
_LABEL_RE = re.compile(
    r'<small class="doc doc-label doc-label-([\w-]+)"><code>([^<]*)</code></small>',
)
_SYMBOL_TOC_RE = re.compile(
    r'<code class="doc-symbol doc-symbol-toc doc-symbol-[\w-]+"></code>&nbsp;'
)
_NAV_LINK_RE = re.compile(
    r'(<a\s+href="#([^"]+)"\s+class="md-nav__link">)(.*?)(</a>)',
    re.DOTALL,
)
_COMPACT_TOC_LABELS = {
    "class-attribute": "var",
    "instance-attribute": "var",
    "module-attribute": "var",
}


def _first_decorator_label(heading_inner: str) -> tuple[str, str] | None:
    match = _LABEL_RE.search(heading_inner)
    if not match:
        return None
    return match.group(1), match.group(2)


def _toc_label_badge(label_class: str, label_text: str) -> str:
    label_text = _COMPACT_TOC_LABELS.get(label_class, label_text)
    return (
        f'<small class="doc doc-label doc-label-toc doc-label-{label_class}">'
        f"<code>{label_text}</code></small>&nbsp;"
    )


def _patch_page_toc_labels(html: str) -> str:
    labels_by_id: dict[str, tuple[str, str]] = {}
    for match in _HEADING_RE.finditer(html):
        label = _first_decorator_label(match.group(3))
        if label:
            labels_by_id[match.group(2)] = label

    if not labels_by_id:
        return html

    def replace_nav(match: re.Match[str]) -> str:
        anchor_id = match.group(2)
        if anchor_id not in labels_by_id:
            return match.group(0)
        label_class, label_text = labels_by_id[anchor_id]
        inner = match.group(3)
        if "doc-symbol-toc" not in inner:
            return match.group(0)
        inner = _SYMBOL_TOC_RE.sub(
            _toc_label_badge(label_class, label_text),
            inner,
            count=1,
        )
        return f"{match.group(1)}{inner}{match.group(4)}"

    return _NAV_LINK_RE.sub(replace_nav, html)


def on_post_build(config: dict[str, Any], **kwargs: Any) -> None:
    site_dir = Path(config["site_dir"])
    for html_path in site_dir.glob("api/**/*.html"):
        text = html_path.read_text(encoding="utf-8")
        patched = _patch_page_toc_labels(text)
        if patched != text:
            html_path.write_text(patched, encoding="utf-8")
