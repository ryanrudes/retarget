# Interactive playground

!!! warning "Local development only"
    Live code talks to a **Jupyter server on your machine**. It does not run on the static docs site or in CI.

On any page with Python or shell examples, use the icon buttons on the top-right of each block: **edit**, **run**, **reset**, **copy**. Edit opens a syntax-highlighted editor; run shows output below. Reset restores the original snippet.

## Setup (two terminals)

**Terminal 1 — docs**

```bash
uv sync --extra dev
uv run mkdocs serve
```

**Terminal 2 — Jupyter** (from the repo root)

```bash
./scripts/docs-jupyter.sh
```

Defaults: token `retarget-docs`, Jupyter at `http://127.0.0.1:8888`, docs at `http://127.0.0.1:8000` (or `http://localhost:8000` — both work). If MkDocs uses another host/port, set `JUPYTER_ALLOW_ORIGIN_PAT` when starting Jupyter.

## Enable

Use the bottom-right control: **Inactive** → flip the switch to **Live** (once per browser session). If Jupyter is not running, the status shows **Uninstalled**.

## Try it

```python
from retarget import TaskKind

[task.value for task in TaskKind]
```

```python
import numpy as np

np.linspace(0.0, 1.0, 5).tolist()
```

```python
from pathlib import Path
import json

path = Path("tests/fixtures/minimal_motion.json")
motion = json.loads(path.read_text())
list(motion.keys())
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Run asks to turn on Live | Flip the switch so the status reads **Live**. |
| Status is **Uninstalled** | Start Jupyter (`./scripts/docs-jupyter.sh`), then click the switch to retry. |
| Could not connect | Start Jupyter via `./scripts/docs-jupyter.sh`. |
| CORS errors / “Uninstalled” | Restart Jupyter via `./scripts/docs-jupyter.sh` (allows both `localhost` and `127.0.0.1`). Open docs at the same host you used before, or set `JUPYTER_ALLOW_ORIGIN_PAT`. |
| Stuck on **Connecting…** | Hard-refresh the docs tab (`Cmd+Shift+R`). Stop Jupyter, run `pkill -f "ipykernel_launcher.*retarget"` to clear orphaned kernels, start `./scripts/docs-jupyter.sh` again, then toggle Live off and on. |
| Jupyter log spam (`Kernel does not exist` / channels 404) | Same as above — usually stale kernels from earlier failed connects. |
| `ModuleNotFoundError: retarget` | Run Jupyter with `uv run` from the repo root after `uv sync --extra dev`. |

API reference signatures are not wired to the kernel.
