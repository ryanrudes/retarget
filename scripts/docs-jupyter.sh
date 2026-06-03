#!/usr/bin/env bash
# Jupyter for docs/interactive-playground.md (Thebe on localhost).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TOKEN="${JUPYTER_TOKEN:-retarget-docs}"
PORT="${JUPYTER_PORT:-8888}"
# MkDocs may be opened as localhost or 127.0.0.1 (different origins). Match any local port.
ALLOW_ORIGIN_PAT="${JUPYTER_ALLOW_ORIGIN_PAT:-^https?://(127\\.0\\.0\\.1|localhost)(:\\d+)?$}"

# Pin kernelspec to this repo's venv (bare "python" can launch the wrong interpreter).
uv run python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (retarget)" >/dev/null

exec uv run jupyter lab \
  --ServerApp.token="$TOKEN" \
  --ServerApp.allow_origin_pat="$ALLOW_ORIGIN_PAT" \
  --ServerApp.allow_credentials=true \
  --ServerApp.open_browser=false \
  --ServerApp.port="$PORT" \
  --ServerApp.root_dir="$ROOT"
