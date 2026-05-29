## Learned User Preferences

- Docs syntax highlighting must match VS Code "Shades of Purple" quality; use Shiki with the official VS Code theme JSON, not Pygments or Highlight.js. Live-code edit mode must use the same Shiki highlighting as view mode (CodeMirror via `codeToTokens`).
- Docs site chrome should follow the Shades of Purple palette instead of Material's gray defaults.
- Main page background should be darker than code block backgrounds so code panels read as elevated.
- TOC sticky title background must match the sidebar color, not the page background.
- Do not show contributors or copyright footer (`copyright` removed from `mkdocs.yml`; `.md-footer { display: none }` in `extra.css`).

## Learned Workspace Facts

- Docs are built with MkDocs Material; preview with `uv sync --extra dev` then `uv run mkdocs serve`.
- Site chrome theming lives in `docs/stylesheets/sop-theme.css`; code block layout overrides in `docs/stylesheets/extra.css`. Wide-layout sidebar backgrounds: paint `.md-sidebar__scrollwrap` and offset Material's `height: 0` plus `.md-main__inner` margin gap.
- Syntax highlighting uses Shiki at runtime via `docs/javascripts/shiki-highlight.mjs` and `docs/themes/shades-of-purple-shiki.json`.
- Live-code execution uses `docs/javascripts/live-code*.mjs` with local Jupyter via `./scripts/docs-jupyter.sh`.
- Dark-mode Shades of Purple palette: page `#1E1E3F`, code blocks `#2D2B55`, sidebars `#222244`.
