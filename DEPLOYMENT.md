# Where things go: local dev vs. Render

Quick reference for what lives where, so it's obvious at a glance instead of
re-deriving it from `config.yaml`/`render.yaml` every time.

## The one-line version

**Local:** everything reads from `config.yaml`. **Render:** the `DATABASE_URL`
environment variable overrides `config.yaml`'s database setting; nothing else
does. Every other setting (area, sensors, model, fusion, mapping) is the same
in both places because it comes from the same file.

---

## Environment variables

| Variable | Where it's set | What it does | If it's missing |
|---|---|---|---|
| `DATABASE_URL` | Render only, auto-injected via `render.yaml`'s `fromDatabase` (points at the `seepsense-db` Postgres instance) | Overrides `backend.database_url` from `config.yaml`. See `backend/config.py: Config.database_url()`. | Falls back to `config.yaml`'s `sqlite:///data/readings.db` — this is what happens locally, and what happens on Render if the env var isn't wired up correctly. |
| `PORT` | Render only, injected automatically by the platform | Which port `uvicorn` binds to (`render.yaml`'s start command uses `--port $PORT`) | N/A locally — you pass `--port 8000` yourself when running uvicorn. |
| `PYTHON_VERSION` | Render only, set in `render.yaml` | Which Python Render's build uses | Render picks a default (may not match 3.11.9). |

Nothing else needs an environment variable. Google Earth Engine auth
(`satellite.gee_project_id`) is a config.yaml value, not an env var, and is
optional everywhere — both local and Render fall back to synthetic satellite
data if it's unset.

## Files that control each environment

| File | Read by | Controls |
|---|---|---|
| `config.yaml` | Both, always | Area, dates, node coordinates, sensor calibration, all thresholds, `backend.database_url` (the *default*, overridden by `DATABASE_URL` on Render) |
| `render.yaml` | Render only, at deploy time | What gets built/deployed: the web service (build/start command, Python version) and the `seepsense-db` Postgres database, plus the env var that links them |
| `requirements.txt` | Both | Python dependencies. `psycopg2-binary` is only exercised when `DATABASE_URL` is a Postgres URL (Render); harmless to have installed locally. |
| `.gitignore` | Neither at runtime — just keeps `data/`, `outputs/`, `.venv/`, `.claude/skills/` out of git | N/A |

## Where your data actually lives

| Environment | Database | Survives a redeploy? |
|---|---|---|
| Local (`.venv\Scripts\python -m uvicorn ...`) | `data/readings.db` (SQLite file on your disk) | Yes — it's your machine, nothing deletes it |
| Render, free plan | `seepsense-db` (managed Postgres), **only if `DATABASE_URL` is correctly set** on the `seepsense` web service | Yes, independent of the web service's own plan/restarts |
| Render, free plan, `DATABASE_URL` missing/broken | Falls back to local SQLite inside the container filesystem | **No** — free plan has no persistent disk; resets on every redeploy/restart |

If `GET /health` on the Render URL shows a `database` field that looks like a
local file path (e.g. `/opt/render/project/src/data/readings.db`) instead of
a Postgres connection, `DATABASE_URL` isn't wired up — see the web service's
**Environment** tab on Render.

## Running each one

**Local:**
```powershell
.venv\Scripts\python -m uvicorn backend.main:app --reload
```
→ http://127.0.0.1:8000/

**Render:** push to `main` — Render redeploys `seepsense` automatically from
`render.yaml`. No manual step beyond `git push`.

## Generated data and outputs (neither environment ships these)

`data/synthetic/`, `data/leakdb/`, `data/satellite.parquet`, and everything
under `outputs/` (trained models, plots, `leak_map.html`) are all
reproducible from the scripts in `scripts/` and are git-ignored on purpose.
Regenerate them locally with `scripts/generate_synthetic.py`,
`scripts/fetch_satellite.py`, `scripts/fetch_leakdb.py`,
`scripts/train_model.py`, and `run_pipeline.py` — never commit their output.
