r"""FastAPI application that receives and serves ESP32 sensor readings.

Endpoints:

* ``POST /ingest``   - validate and store one reading from a node.
* ``GET  /readings`` - fetch stored readings, filtered by node and time range.
* ``GET  /nodes``    - list the nodes configured in ``config.yaml``.
* ``GET  /health``   - liveness check plus a row count, useful while testing.

Run it with::

    .venv\Scripts\python -m uvicorn backend.main:app --reload

Interactive documentation is then available at ``/docs``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.config import Config, get_config
from backend.db import get_session, init_db
from backend.models import Reading
from backend.schemas import (
    HealthResponse,
    IngestResponse,
    ReadingIn,
    ReadingOut,
    ReadingsResponse,
    to_naive_utc,
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the database file and tables before the first request is served.

    Args:
        app: The FastAPI application (unused, required by the interface).

    Yields:
        Control back to the server for the lifetime of the application.
    """
    init_db()
    yield


app = FastAPI(
    title="Leak Detection Sensor API",
    description=(
        "Ingestion endpoint for ESP32 ground sensor nodes in the "
        "satellite-assisted pipe leak detection prototype."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _config() -> Config:
    """FastAPI dependency returning the cached project configuration."""
    return get_config()


@app.get("/", tags=["ui"])
def dashboard() -> FileResponse:
    """Serve the dashboard's single HTML page (backend/static/index.html)."""
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(session: Session = Depends(get_session)) -> HealthResponse:
    """Report that the API is up and how many readings are stored.

    Args:
        session: Injected database session.

    Returns:
        Service status, the database file in use and the current row count.
    """
    count = session.scalar(select(func.count()).select_from(Reading)) or 0
    return HealthResponse(
        status="ok",
        database=str(get_config().database_path()),
        reading_count=int(count),
    )


@app.get("/nodes", tags=["meta"])
def list_nodes(config: Config = Depends(_config)) -> list[dict]:
    """Return the sensor nodes declared in ``config.yaml``.

    The map and fusion layers need node coordinates, and the fake-payload
    script needs the node IDs, so the configured list is exposed here rather
    than duplicated in several places.

    Args:
        config: Injected project configuration.

    Returns:
        A list of node dictionaries with ``node_id``, ``lat`` and ``lon``.
    """
    return list(config.get("nodes"))


@app.get("/map", tags=["meta"])
def get_leak_map(config: Config = Depends(_config)) -> FileResponse:
    """Serve the Phase 8 Folium leak-probability map as a web page.

    Args:
        config: Injected project configuration.

    Returns:
        The saved ``mapping.output_path`` HTML file.

    Raises:
        HTTPException: 404 if the map hasn't been generated yet (run
            ``run_pipeline.py`` first).
    """
    path = config.resolve_path(config.get("mapping.output_path"))
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No map found at {path}. Run run_pipeline.py first.",
        )
    return FileResponse(path, media_type="text/html")


@app.get("/api/dashboard", tags=["ui"])
def dashboard_data(
    session: Session = Depends(get_session),
    config: Config = Depends(_config),
) -> dict:
    """Aggregate everything the dashboard page needs into one response.

    Combines: live reading count, each node's most recent stored reading,
    whether a trained model/leak map exist yet, and (if `run_pipeline.py`
    has been run) the last fusion run's per-node leak probabilities.

    Args:
        session: Injected database session.
        config: Injected project configuration.

    Returns:
        A dict with ``reading_count``, ``nodes`` (config + latest reading +
        risk, where available), ``model`` (metrics, if trained) and
        ``map_available``.
    """
    reading_count = int(session.scalar(select(func.count()).select_from(Reading)) or 0)

    latest_by_node: dict[str, dict] = {}
    for node_id, in session.execute(select(Reading.node_id).distinct()):
        row = session.scalars(
            select(Reading).where(Reading.node_id == node_id).order_by(desc(Reading.timestamp)).limit(1)
        ).first()
        if row is not None:
            # Trim the accel burst: the dashboard only shows scalar fields,
            # and a full 256-sample burst per node would bloat this response.
            reading = row.to_dict()
            reading.pop("accel", None)
            latest_by_node[node_id] = reading

    summary_path = config.resolve_path(config.get("mapping.output_path")).parent / "pipeline_summary.json"
    pipeline_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    risk_by_node = {n["node_id"]: n for n in (pipeline_summary or {}).get("nodes", [])}

    nodes = []
    for node in config.get("nodes"):
        node_id = node["node_id"]
        nodes.append(
            {
                **node,
                "latest_reading": latest_by_node.get(node_id),
                "leak_probability": risk_by_node.get(node_id, {}).get("leak_probability"),
            }
        )

    metrics_path = config.resolve_path(config.get("model.output_dir")) / "metrics.json"
    model_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None

    leakdb_metrics_path = config.resolve_path(config.get("leakdb.metrics_path"))
    leakdb_metrics = json.loads(leakdb_metrics_path.read_text(encoding="utf-8")) if leakdb_metrics_path.exists() else None

    map_path = config.resolve_path(config.get("mapping.output_path"))

    return {
        "reading_count": reading_count,
        "nodes": nodes,
        "model": model_metrics,
        "leakdb": leakdb_metrics,
        "pipeline": pipeline_summary,
        "map_available": map_path.exists(),
    }


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingest"],
)
def ingest_reading(
    reading: ReadingIn,
    session: Session = Depends(get_session),
) -> IngestResponse:
    """Validate one ESP32 payload and store it.

    FastAPI has already validated the body against :class:`ReadingIn` by the
    time this function runs, so a malformed payload never reaches the
    database and the node receives HTTP 422 with a field-level explanation.

    Args:
        reading: The validated payload.
        session: Injected database session.

    Returns:
        The primary key and identifying fields of the stored reading.

    Raises:
        HTTPException: 500 if the row could not be written.
    """
    row = Reading(**reading.to_orm_kwargs())
    session.add(row)
    try:
        session.commit()
    except Exception as exc:  # pragma: no cover - defensive, hard to trigger
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store reading: {exc}",
        ) from exc
    session.refresh(row)

    return IngestResponse(id=row.id, node_id=row.node_id, timestamp=row.timestamp)


@app.get("/readings", response_model=ReadingsResponse, tags=["query"])
def get_readings(
    session: Session = Depends(get_session),
    config: Config = Depends(_config),
    node_id: str | None = Query(
        default=None, description="Only return readings from this node."
    ),
    start: datetime | None = Query(
        default=None,
        description="Inclusive lower bound on the reading timestamp (ISO-8601).",
    ),
    end: datetime | None = Query(
        default=None,
        description="Inclusive upper bound on the reading timestamp (ISO-8601).",
    ),
    limit: int | None = Query(
        default=None,
        ge=1,
        description="Maximum rows to return; defaults to backend.max_readings_per_query.",
    ),
    offset: int = Query(default=0, ge=0, description="Rows to skip, for paging."),
) -> ReadingsResponse:
    """Return stored readings, oldest first, filtered by node and time range.

    Results are ordered by timestamp because every downstream phase treats the
    readings as a time series; a stable chronological order means callers do
    not have to re-sort.

    Args:
        session: Injected database session.
        config: Injected project configuration.
        node_id: Optional node filter.
        start: Optional inclusive start of the time range.
        end: Optional inclusive end of the time range.
        limit: Optional row cap, clamped to the configured maximum.
        offset: Number of rows to skip.

    Returns:
        The matching readings together with the paging parameters used.

    Raises:
        HTTPException: 400 if ``start`` is later than ``end``.
    """
    max_rows = int(config.get("backend.max_readings_per_query"))
    effective_limit = max_rows if limit is None else min(limit, max_rows)

    # Incoming query parameters may carry a timezone; the stored column does
    # not, so normalise before comparing.
    start_utc = to_naive_utc(start) if start is not None else None
    end_utc = to_naive_utc(end) if end is not None else None

    if start_utc is not None and end_utc is not None and start_utc > end_utc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'start' must not be later than 'end'",
        )

    query = select(Reading)
    if node_id is not None:
        query = query.where(Reading.node_id == node_id)
    if start_utc is not None:
        query = query.where(Reading.timestamp >= start_utc)
    if end_utc is not None:
        query = query.where(Reading.timestamp <= end_utc)

    query = query.order_by(Reading.timestamp, Reading.id).offset(offset).limit(effective_limit)
    rows = session.scalars(query).all()

    readings = [ReadingOut.from_orm_row(row) for row in rows]
    return ReadingsResponse(
        count=len(readings),
        limit=effective_limit,
        offset=offset,
        readings=readings,
    )
