"""Phase 2 tests: ingestion, validation and querying.

These run against the real FastAPI application and real SQLAlchemy code; only
the database file is swapped for a temporary one (see ``conftest.py``).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.config import Config
from scripts.post_fake_payloads import make_invalid_payload, make_payload

# A fixed instant, so assertions about time ranges are deterministic.
BASE_TIME = datetime(2025, 2, 1, 12, 0, 0)


def valid_payload(
    config: Config,
    node_id: str = "node_01",
    timestamp: datetime | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a valid payload for tests.

    Args:
        config: Project configuration.
        node_id: Node the payload claims to come from.
        timestamp: Measurement time; defaults to :data:`BASE_TIME`.
        seed: Seed for the noise, so payloads are reproducible.

    Returns:
        A payload dictionary that should be accepted by ``POST /ingest``.
    """
    return make_payload(
        node_id=node_id,
        timestamp=timestamp if timestamp is not None else BASE_TIME,
        config=config,
        rng=random.Random(seed),
    )


# --- POST /ingest, valid payloads ------------------------------------------


def test_ingest_accepts_a_valid_payload(client: TestClient, config: Config) -> None:
    """A well-formed payload is stored and acknowledged with HTTP 201."""
    response = client.post("/ingest", json=valid_payload(config))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] > 0
    assert body["node_id"] == "node_01"
    assert body["stored"] is True


def test_ingested_reading_is_returned_unchanged(client: TestClient, config: Config) -> None:
    """Every field survives the round trip, including the accelerometer burst."""
    payload = valid_payload(config)
    assert client.post("/ingest", json=payload).status_code == 201

    readings = client.get("/readings").json()["readings"]
    assert len(readings) == 1
    stored = readings[0]

    assert stored["soil_raw"] == payload["soil_raw"]
    assert stored["temp_c"] == pytest.approx(payload["temp_c"])
    assert stored["humidity_pct"] == pytest.approx(payload["humidity_pct"])
    assert stored["pressure_hpa"] == pytest.approx(payload["pressure_hpa"])
    assert stored["mag"] == pytest.approx(payload["mag"])
    assert len(stored["accel"]) == len(payload["accel"])
    assert stored["accel"][0] == pytest.approx(payload["accel"][0])
    assert stored["accel"][-1] == pytest.approx(payload["accel"][-1])


def test_accel_burst_length_matches_configuration(client: TestClient, config: Config) -> None:
    """The stored burst has the number of samples the configuration declares."""
    client.post("/ingest", json=valid_payload(config))
    stored = client.get("/readings").json()["readings"][0]
    assert len(stored["accel"]) == config.get("sensor.accel_burst_samples")


def test_timezone_aware_timestamps_are_normalised_to_utc(
    client: TestClient, config: Config
) -> None:
    """A payload timestamped in +05:30 is stored as the equivalent UTC time."""
    local = datetime(2025, 2, 1, 17, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    payload = valid_payload(config)
    payload["timestamp"] = local.isoformat()

    assert client.post("/ingest", json=payload).status_code == 201

    stored = client.get("/readings").json()["readings"][0]
    # 17:30 +05:30 is 12:00 UTC.
    assert stored["timestamp"].startswith("2025-02-01T12:00:00")


def test_generated_fake_payloads_all_validate(client: TestClient, config: Config) -> None:
    """The fake-payload generator never produces something the API rejects."""
    rng = random.Random(config.get("project.random_seed"))
    for step in range(10):
        payload = make_payload(
            node_id="node_02",
            timestamp=BASE_TIME + timedelta(minutes=step),
            config=config,
            rng=rng,
        )
        assert client.post("/ingest", json=payload).status_code == 201


# --- POST /ingest, invalid payloads ----------------------------------------


def test_ingest_rejects_the_known_bad_payload(client: TestClient) -> None:
    """The deliberately invalid payload used by the script is rejected."""
    response = client.post("/ingest", json=make_invalid_payload())
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, bad_value, reason",
    [
        ("soil_raw", 99999, "ADC value above the configured maximum"),
        ("soil_raw", -1, "negative ADC value"),
        ("soil_raw", "wet", "ADC value is not an integer"),
        ("humidity_pct", 150.0, "humidity above 100 percent"),
        ("pressure_hpa", 10.0, "implausible pressure"),
        ("temp_c", 500.0, "implausible temperature"),
        ("mag", [1.0, 2.0], "magnetometer vector missing an axis"),
        ("accel", [], "empty accelerometer burst"),
        ("accel", [[0.1, 0.2]], "accelerometer sample missing an axis"),
        ("accel", [[0.0, 0.0, 999.0]], "accelerometer value out of range"),
        ("timestamp", "not-a-date", "unparseable timestamp"),
        ("node_id", "", "empty node id"),
    ],
)
def test_ingest_rejects_invalid_field(
    client: TestClient,
    config: Config,
    field: str,
    bad_value: Any,
    reason: str,
) -> None:
    """Each individually broken field produces HTTP 422 and stores nothing."""
    payload = valid_payload(config)
    payload[field] = bad_value

    response = client.post("/ingest", json=payload)

    assert response.status_code == 422, "should have been rejected: {}".format(reason)
    assert client.get("/readings").json()["count"] == 0


@pytest.mark.parametrize("missing_field", ["node_id", "timestamp", "soil_raw", "accel", "mag"])
def test_ingest_rejects_missing_field(
    client: TestClient, config: Config, missing_field: str
) -> None:
    """A payload with a required field removed is rejected."""
    payload = valid_payload(config)
    del payload[missing_field]
    assert client.post("/ingest", json=payload).status_code == 422


def test_ingest_rejects_unknown_field(client: TestClient, config: Config) -> None:
    """An unexpected extra field is rejected rather than silently dropped."""
    payload = valid_payload(config)
    payload["battery_v"] = 3.7
    assert client.post("/ingest", json=payload).status_code == 422


def test_ingest_rejects_oversized_burst(client: TestClient, config: Config) -> None:
    """A burst longer than the configured maximum is refused."""
    payload = valid_payload(config)
    too_many = int(config.get("backend.max_accel_burst_samples")) + 1
    payload["accel"] = [[0.0, 0.0, 1.0]] * too_many
    assert client.post("/ingest", json=payload).status_code == 422


# --- GET /readings ---------------------------------------------------------


@pytest.fixture()
def populated_client(client: TestClient, config: Config) -> TestClient:
    """Return a client whose database holds a known set of readings.

    Two nodes post one reading per minute for five minutes starting at
    :data:`BASE_TIME`, giving ten rows in total.
    """
    rng = random.Random(1)
    for step in range(5):
        timestamp = BASE_TIME + timedelta(minutes=step)
        for node_id in ("node_01", "node_02"):
            payload = make_payload(node_id, timestamp, config, rng)
            assert client.post("/ingest", json=payload).status_code == 201
    return client


def test_readings_returns_everything_by_default(populated_client: TestClient) -> None:
    """With no filters, all stored readings come back."""
    body = populated_client.get("/readings").json()
    assert body["count"] == 10
    assert len(body["readings"]) == 10


def test_readings_are_ordered_by_time(populated_client: TestClient) -> None:
    """Readings come back oldest first, as a time series."""
    timestamps = [r["timestamp"] for r in populated_client.get("/readings").json()["readings"]]
    assert timestamps == sorted(timestamps)


def test_readings_filter_by_node(populated_client: TestClient) -> None:
    """The node filter returns only that node's readings."""
    body = populated_client.get("/readings", params={"node_id": "node_01"}).json()
    assert body["count"] == 5
    assert {r["node_id"] for r in body["readings"]} == {"node_01"}


def test_readings_filter_by_time_range(populated_client: TestClient) -> None:
    """The time range is inclusive at both ends."""
    start = (BASE_TIME + timedelta(minutes=1)).isoformat()
    end = (BASE_TIME + timedelta(minutes=3)).isoformat()

    body = populated_client.get("/readings", params={"start": start, "end": end}).json()

    # Minutes 1, 2 and 3 from two nodes.
    assert body["count"] == 6
    for reading in body["readings"]:
        assert start <= reading["timestamp"] <= end


def test_readings_filter_by_node_and_time(populated_client: TestClient) -> None:
    """Node and time filters combine."""
    params = {
        "node_id": "node_02",
        "start": (BASE_TIME + timedelta(minutes=2)).isoformat(),
    }
    body = populated_client.get("/readings", params=params).json()

    assert body["count"] == 3
    assert {r["node_id"] for r in body["readings"]} == {"node_02"}


def test_readings_filter_accepts_timezone_aware_bounds(populated_client: TestClient) -> None:
    """Query bounds may carry a timezone; they are converted to UTC first."""
    # 17:31 +05:30 is 12:01 UTC, so minutes 1 to 4 from two nodes remain.
    start = datetime(2025, 2, 1, 17, 31, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    body = populated_client.get("/readings", params={"start": start.isoformat()}).json()
    assert body["count"] == 8


def test_readings_empty_range_returns_nothing(populated_client: TestClient) -> None:
    """A range with no readings in it returns an empty list, not an error."""
    body = populated_client.get("/readings", params={"start": "2030-01-01T00:00:00"}).json()
    assert body["count"] == 0
    assert body["readings"] == []


def test_readings_reject_inverted_range(populated_client: TestClient) -> None:
    """A start later than the end is a client error, not an empty result."""
    params = {"start": "2025-02-02T00:00:00", "end": "2025-02-01T00:00:00"}
    assert populated_client.get("/readings", params=params).status_code == 400


def test_readings_limit_and_offset(populated_client: TestClient) -> None:
    """Paging returns disjoint, consecutive slices of the time series."""
    first = populated_client.get("/readings", params={"limit": 4}).json()
    second = populated_client.get("/readings", params={"limit": 4, "offset": 4}).json()

    assert first["count"] == 4
    assert second["count"] == 4
    assert {r["id"] for r in first["readings"]}.isdisjoint({r["id"] for r in second["readings"]})


def test_readings_limit_is_clamped_to_configured_maximum(
    populated_client: TestClient, config: Config
) -> None:
    """A caller cannot ask for more rows than the configured cap."""
    max_rows = int(config.get("backend.max_readings_per_query"))
    body = populated_client.get("/readings", params={"limit": max_rows + 1000}).json()
    assert body["limit"] == max_rows


def test_readings_reject_invalid_paging(populated_client: TestClient) -> None:
    """Zero or negative paging values are refused."""
    assert populated_client.get("/readings", params={"limit": 0}).status_code == 422
    assert populated_client.get("/readings", params={"offset": -1}).status_code == 422


# --- meta endpoints --------------------------------------------------------


def test_health_reports_row_count(client: TestClient, config: Config) -> None:
    """The health endpoint counts stored readings."""
    assert client.get("/health").json()["reading_count"] == 0
    client.post("/ingest", json=valid_payload(config))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["reading_count"] == 1


def test_nodes_endpoint_matches_configuration(client: TestClient, config: Config) -> None:
    """The nodes endpoint exposes exactly what config.yaml declares."""
    nodes = client.get("/nodes").json()
    assert [node["node_id"] for node in nodes] == config.node_ids()
    for node in nodes:
        assert "lat" in node and "lon" in node
