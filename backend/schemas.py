"""Pydantic models describing the ESP32 payload and the API responses.

These schemas are the contract between the firmware and the backend.  Every
bound used here (ADC range, plausible temperature, maximum burst length) comes
from ``config.yaml`` rather than being written into the code, so the same
backend can be pointed at different hardware by editing configuration only.

A payload that fails validation is rejected with HTTP 422 and never reaches
the database, which keeps obviously broken readings out of the feature
pipeline downstream.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.config import get_config

# --- Bounds pulled from configuration at import time ------------------------
_cfg = get_config()

SOIL_RAW_MIN: int = int(_cfg.get("sensor.soil_raw_min"))
SOIL_RAW_MAX: int = int(_cfg.get("sensor.soil_raw_max"))
MAX_ACCEL_SAMPLES: int = int(_cfg.get("backend.max_accel_burst_samples"))

TEMP_MIN, TEMP_MAX = (float(v) for v in _cfg.get("sensor.valid_range.temp_c"))
HUMIDITY_MIN, HUMIDITY_MAX = (float(v) for v in _cfg.get("sensor.valid_range.humidity_pct"))
PRESSURE_MIN, PRESSURE_MAX = (float(v) for v in _cfg.get("sensor.valid_range.pressure_hpa"))
MAG_MIN, MAG_MAX = (float(v) for v in _cfg.get("sensor.valid_range.mag_ut"))
ACCEL_MIN, ACCEL_MAX = (float(v) for v in _cfg.get("sensor.valid_range.accel_g"))


def to_naive_utc(value: datetime) -> datetime:
    """Convert a datetime to naive UTC for storage in SQLite.

    Timestamps arriving from a node may be timezone-aware (``...+05:30``) or
    naive.  Naive values are assumed to already be UTC, which is what the
    firmware is expected to send.

    Args:
        value: The datetime to normalise.

    Returns:
        The same instant expressed in UTC with ``tzinfo`` removed.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


class ReadingIn(BaseModel):
    """One sensor reading as posted by an ESP32 node to ``POST /ingest``."""

    # Reject unknown fields: a typo in the firmware should be a loud 422, not a
    # silently dropped column.
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Identifier of the node, e.g. 'node_01'.",
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement time in ISO-8601 format. Naive values are treated as UTC.",
    )
    soil_raw: int = Field(
        ...,
        ge=SOIL_RAW_MIN,
        le=SOIL_RAW_MAX,
        description="Raw capacitive soil probe ADC count; lower means wetter.",
    )
    accel: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=MAX_ACCEL_SAMPLES,
        description="Burst of [ax, ay, az] samples in g, captured at the node's sample rate.",
    )
    mag: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Magnetometer vector [mx, my, mz] in microtesla.",
    )
    temp_c: float = Field(..., ge=TEMP_MIN, le=TEMP_MAX, description="Air temperature in C.")
    humidity_pct: float = Field(
        ..., ge=HUMIDITY_MIN, le=HUMIDITY_MAX, description="Relative humidity in percent."
    )
    pressure_hpa: float = Field(
        ..., ge=PRESSURE_MIN, le=PRESSURE_MAX, description="Barometric pressure in hPa."
    )

    @field_validator("timestamp")
    @classmethod
    def _normalise_timestamp(cls, value: datetime) -> datetime:
        """Store every timestamp as naive UTC."""
        return to_naive_utc(value)

    @field_validator("accel")
    @classmethod
    def _check_accel_samples(cls, value: list[list[float]]) -> list[list[float]]:
        """Check that every sample is a 3-axis vector within the sensor range.

        Args:
            value: The accelerometer burst.

        Returns:
            The burst unchanged.

        Raises:
            ValueError: If a sample has the wrong length or is out of range.
        """
        for index, sample in enumerate(value):
            if len(sample) != 3:
                raise ValueError(
                    f"accel sample {index} has {len(sample)} components, expected 3"
                )
            for axis_value in sample:
                if not ACCEL_MIN <= axis_value <= ACCEL_MAX:
                    raise ValueError(
                        f"accel sample {index} value {axis_value} outside "
                        f"[{ACCEL_MIN}, {ACCEL_MAX}] g"
                    )
        return value

    @field_validator("mag")
    @classmethod
    def _check_mag_range(cls, value: list[float]) -> list[float]:
        """Check that each magnetometer component is within the sensor range."""
        for axis_value in value:
            if not MAG_MIN <= axis_value <= MAG_MAX:
                raise ValueError(
                    f"mag value {axis_value} outside [{MAG_MIN}, {MAG_MAX}] uT"
                )
        return value

    def to_orm_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for constructing a ``Reading`` ORM row.

        Returns:
            A mapping with the accelerometer burst serialised to JSON and the
            magnetometer vector split into its three columns.
        """
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "soil_raw": self.soil_raw,
            "accel_json": json.dumps(self.accel),
            "accel_sample_count": len(self.accel),
            "mag_x": self.mag[0],
            "mag_y": self.mag[1],
            "mag_z": self.mag[2],
            "temp_c": self.temp_c,
            "humidity_pct": self.humidity_pct,
            "pressure_hpa": self.pressure_hpa,
        }


class ReadingOut(BaseModel):
    """One stored reading as returned by ``GET /readings``."""

    id: int
    node_id: str
    timestamp: datetime
    received_at: datetime
    soil_raw: int
    accel: list[list[float]]
    mag: list[float]
    temp_c: float
    humidity_pct: float
    pressure_hpa: float

    @classmethod
    def from_orm_row(cls, row: Any) -> "ReadingOut":
        """Build a response model from a :class:`backend.models.Reading` row.

        Args:
            row: The ORM row to convert.

        Returns:
            The corresponding response model.
        """
        return cls(
            id=row.id,
            node_id=row.node_id,
            timestamp=row.timestamp,
            received_at=row.received_at,
            soil_raw=row.soil_raw,
            accel=row.accel,
            mag=row.mag,
            temp_c=row.temp_c,
            humidity_pct=row.humidity_pct,
            pressure_hpa=row.pressure_hpa,
        )


class IngestResponse(BaseModel):
    """Acknowledgement returned by ``POST /ingest``."""

    id: int = Field(..., description="Primary key of the stored reading.")
    node_id: str
    timestamp: datetime
    stored: bool = Field(True, description="Always True; failures raise an HTTP error.")


class ReadingsResponse(BaseModel):
    """Envelope returned by ``GET /readings``."""

    count: int = Field(..., description="Number of readings in this response.")
    limit: int = Field(..., description="Maximum number of readings requested.")
    offset: int = Field(..., description="Number of readings skipped.")
    readings: list[ReadingOut]


class HealthResponse(BaseModel):
    """Payload returned by ``GET /health``."""

    status: str
    database: str
    reading_count: int
