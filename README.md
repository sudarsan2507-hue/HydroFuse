# Satellite-Assisted Underground Water Pipe Leak Detection

A research prototype that fuses **ground sensor readings** (ESP32 nodes: soil moisture,
vibration, magnetometer, temperature/humidity/pressure) with **satellite observations**
(Sentinel-1, Sentinel-2, SMAP, Copernicus DEM, MODIS LST) to produce a **leak-probability
map** for a small test area.

The hardware is not built yet, so the whole pipeline runs on synthetic data that follows the
exact schema the firmware will send. Switching to real data is a change of `--source`, not a
change of code.

---

## Status

| Phase | Component | State |
|-------|-----------|-------|
| 1 | Project scaffold, `config.yaml`, venv | **done** |
| 2 | FastAPI backend, SQLite storage, fake-payload script | **done** |
| 3 | Synthetic scenario generator (no_leak / leak / rain_not_leak) | **done** |
| 4 | Google Earth Engine satellite pipeline | not started |
| 5 | Feature engineering (FFT, moisture, environment) | not started |
| 6 | Leak classifier (RandomForest vs XGBoost) | not started |
| 7 | Fusion layer (node-corrected SMAP + IDW) | not started |
| 8 | Folium leak-probability map | not started |
| 9 | `run_pipeline.py` one-command runner | not started |

---

## Setup

Requires **Python 3.11**.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

All later commands assume `.venv\Scripts\python`; activating the venv
(`.venv\Scripts\Activate.ps1`) and typing `python` works just as well.

---

## Configuration

Everything tunable lives in [config.yaml](config.yaml): the test-area bounding box, the study
date range, node coordinates, soil-probe calibration, the leak frequency band, model
hyper-parameters, and the alert threshold for the map. **No module hardcodes these values** —
they are read through `backend.config.load_config()`:

```python
from backend.config import load_config

cfg = load_config()
cfg.get("area.center_lat")            # dotted lookup
cfg.get("mapping.alert_probability_threshold")
cfg.resolve_path("data/satellite.parquet")   # absolute, relative to the project root
```

A missing key raises `ConfigError` naming the key, rather than a `KeyError` deep in the
pipeline.

Values you are most likely to change first:

- `area.*` and `date_range.*` — where and when
- `nodes` — the physical node IDs and coordinates (`node_id` must match the firmware)
- `sensor.soil_raw_dry` / `sensor.soil_raw_wet` — probe calibration, measured in air and water
- `synthetic.leak_band_hz` — the frequency band a pressurised leak shows up in
- `mapping.alert_probability_threshold` — when a grid cell gets a red alert marker

---

## Phase 2: the sensor backend

### Run the API

```powershell
.venv\Scripts\python -m uvicorn backend.main:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>

The SQLite database is created automatically at `data/readings.db` (path from
`backend.database_url`).

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/ingest` | Validate and store one ESP32 payload. `201` on success, `422` with field-level detail on a bad payload. |
| `GET` | `/readings` | Stored readings, oldest first. Filters: `node_id`, `start`, `end`, `limit`, `offset`. |
| `GET` | `/nodes` | The nodes declared in `config.yaml`, with coordinates. |
| `GET` | `/health` | Liveness check plus the current row count. |

### Payload format

This is the contract the ESP32 firmware must follow:

```json
{
  "node_id": "node_01",
  "timestamp": "2025-02-01T12:00:00Z",
  "soil_raw": 2250,
  "accel": [[0.01, -0.02, 1.00], [0.00, 0.01, 0.99]],
  "mag": [22.1, -8.3, 38.4],
  "temp_c": 24.6,
  "humidity_pct": 58.2,
  "pressure_hpa": 1010.4
}
```

Notes:

- `timestamp` is ISO-8601. A timezone offset is accepted and converted; a naive timestamp is
  assumed to be UTC. Everything is stored as naive UTC.
- `soil_raw` is the raw ADC count, **lower means wetter**. Bounds come from
  `sensor.soil_raw_min` / `sensor.soil_raw_max`.
- `accel` is the burst of `[ax, ay, az]` samples in g at `sensor.accel_sample_rate_hz`. It is
  stored verbatim as JSON in one column, because Phase 5 always reads the burst as a whole.
- Unknown fields are **rejected**, so a firmware typo is a loud `422` rather than a silently
  dropped column.

### Test it without hardware

With the API running in one terminal:

```powershell
# 30 readings from every configured node
.venv\Scripts\python scripts\post_fake_payloads.py --count 30

# look at a payload without sending it
.venv\Scripts\python scripts\post_fake_payloads.py --count 1 --dry-run

# check that a bad payload is properly rejected (expect HTTP 422)
.venv\Scripts\python scripts\post_fake_payloads.py --invalid

# simulate a live node: one reading every 2 real seconds
.venv\Scripts\python scripts\post_fake_payloads.py --count 100 --delay 2
```

Then query them back:

```powershell
curl "http://127.0.0.1:8000/readings?node_id=node_01&limit=5"
```

This script produces *plausible* readings only — it is a smoke test for the API. The
physically realistic, **labelled** scenario generator is Phase 3, described next.

---

## Phase 3: synthetic scenario generator

Generates labelled time series for three scenarios, in the exact JSON schema `POST /ingest`
accepts, so the rest of the pipeline can be built and tested before any hardware exists.

### Run it

```powershell
.venv\Scripts\python scripts\generate_synthetic.py
```

This writes:

- `data/synthetic/no_leak.parquet`, `leak.parquet`, `rain_not_leak.parquet` — one labelled
  DataFrame per scenario
- `data/synthetic/all_scenarios.parquet` — all three concatenated, with a `scenario` column
- `outputs/synthetic_soil_plot.png` — soil moisture over time for all three scenarios
- a console summary: mean soil moisture, accelerometer vibration RMS, and the dominant
  vibration frequency per scenario, so the leak's signature is visible before Phase 5 builds
  features on top of it

### Scenarios

| Scenario | `label` | `rain_flag` | Soil moisture | Vibration | Environment |
|----------|:-:|:-:|---|---|---|
| `no_leak` | 0 | False | Stable, daily sinusoidal drift | Background noise only | Normal daily cycle |
| `leak` | 1 | False | Steady linear drift, wetter over time | Noise **+ continuous tone** at the centre of `synthetic.leak_band_hz` | Normal — this is what tells a leak apart from rain |
| `rain_not_leak` | 0 | True | Sharp spike(s), exponential decay | Background noise only | Humidity jumps, pressure dips *before* the spike, slight temp dip |

Every scenario also gets a slowly random-walking magnetometer, footstep-like transient
vibration spikes (`synthetic.transient_spike_probability`), and simulated sensor dropouts
(`synthetic.dropout_probability`, default 2%): one or more fields replaced with `None` to
simulate a corrupted transmission. A dropout row is intentionally **not** valid against
`backend.schemas.ReadingIn` — a corrupted payload should be rejected the same way a real one
would be — so it carries a `dropout` column instead of being silently interpolated. Every
other row validates against the real ingest schema; `tests/test_synthetic.py` checks both
halves of that contract.

Every number above — durations, baselines, drift rates, noise levels, event amplitudes, the
leak band — is read from `config.yaml`'s `synthetic` section; nothing is hardcoded in
`synthetic/`.

### Module layout

- `synthetic/scenarios.py` — pure physical models (no config, no I/O): daily cycles, soil
  drift/spikes, environment, magnetometer walk, accelerometer bursts.
- `synthetic/dropout.py` — simulated field-level transmission failures.
- `synthetic/analysis.py` — FFT/RMS helpers used by both the sanity-check script and the tests.
- `synthetic/generator.py` — reads `config.yaml` and wires the pure functions above into a
  labelled DataFrame per scenario.
- `synthetic/io.py` — parquet save/load (accel/mag are JSON-encoded for storage, decoded back
  to Python lists on load) and `row_to_payload()`, which turns a DataFrame row back into the
  same dict shape `POST /ingest` expects.

### Tests

```powershell
.venv\Scripts\python -m pytest tests\test_synthetic.py
```

Covers: every clean record validates against the real `ReadingIn` schema and every
dropout-affected record correctly fails it; the leak scenario's dominant accelerometer
frequency falls inside the configured band with a strong, persistent peak; no_leak and
rain_not_leak show no persistent vibration tone; rain produces a moisture spike more than 2x
the normal daily swing; the leak scenario trends wetter over time; identical seeds reproduce
identical output (including the accelerometer bursts) and different seeds don't collide;
parquet round-trips preserve schema and dropout flags; and changing a configured value (soil
baseline, leak band) measurably changes the generated data.

---

## Tests

```powershell
.venv\Scripts\python -m pytest
```

Each test gets its own temporary SQLite file, so the tests never touch `data/readings.db` and
never see each other's rows.

---

## Project layout

```
backend/     FastAPI app, SQLAlchemy models, Pydantic schemas, config loader
satellite/   Phase 4: Google Earth Engine pulls -> data/satellite.parquet
synthetic/   Phase 3: labelled scenario generator
features/    Phase 5: FFT / moisture / environment feature functions
model/       Phase 6: classifier training and evaluation
fusion/      Phase 7: node-corrected SMAP + inverse-distance weighting
mapping/     Phase 8: Folium map generation
scripts/     Command-line helpers (fake payload poster)
tests/       pytest suite, one module per phase
data/        SQLite database and cached satellite frames (git-ignored)
outputs/     Trained models, plots, leak_map.html (git-ignored)
notebooks/   Exploration
```

---

## Dependencies

Standard scientific Python (numpy, pandas, scipy, matplotlib, scikit-learn) plus the libraries
named in the project spec: FastAPI/Pydantic/SQLAlchemy, earthengine-api + geemap, xgboost,
shap, folium, joblib. Supporting packages: `PyYAML` (reads `config.yaml`), `uvicorn` (runs
FastAPI), `httpx` (the poster script and FastAPI's test client), `pyarrow` (parquet).

Google Earth Engine is free but needs a Google account and a registered cloud project; put the
project ID in `satellite.gee_project_id`. If it is not set up, Phase 4 falls back to a small
synthetic satellite frame so the rest of the pipeline still runs. **No paid service is used.**
