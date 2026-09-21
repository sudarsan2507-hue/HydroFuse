"""SQLAlchemy ORM models for stored sensor readings.

One row in :class:`Reading` corresponds to exactly one JSON payload posted by
an ESP32 node.  The accelerometer burst is stored as a JSON string in a single
text column: it is a variable-length array that we always read back as a whole,
so a separate samples table would add joins without adding value.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base class for all ORM models in this project."""


def utcnow() -> datetime:
    """Return the current UTC time as a naive datetime.

    SQLite has no timezone-aware datetime type, so every timestamp in the
    database is stored as naive UTC.  Conversion happens at the edges (see
    :func:`backend.schemas.to_naive_utc`).

    Returns:
        The current time in UTC, with ``tzinfo`` stripped.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Reading(Base):
    """A single sensor reading posted by one node.

    Attributes:
        id: Surrogate primary key.
        node_id: Identifier of the node that produced the reading.
        timestamp: Measurement time reported by the node, naive UTC.
        received_at: Server time the reading was stored, naive UTC.  Useful for
            spotting nodes whose clock has drifted.
        soil_raw: Raw capacitive soil probe reading (lower = wetter).
        accel_json: JSON-encoded list of ``[ax, ay, az]`` samples.
        accel_sample_count: Number of samples in the burst, denormalised so
            that queries can filter on burst size without parsing the JSON.
        mag_x, mag_y, mag_z: Magnetometer components.
        temp_c: Air temperature in degrees Celsius.
        humidity_pct: Relative humidity in percent.
        pressure_hpa: Barometric pressure in hectopascals.
    """

    __tablename__ = "readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    node_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )

    soil_raw: Mapped[int] = mapped_column(Integer, nullable=False)

    accel_json: Mapped[str] = mapped_column(Text, nullable=False)
    accel_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)

    mag_x: Mapped[float] = mapped_column(Float, nullable=False)
    mag_y: Mapped[float] = mapped_column(Float, nullable=False)
    mag_z: Mapped[float] = mapped_column(Float, nullable=False)

    temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    humidity_pct: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_hpa: Mapped[float] = mapped_column(Float, nullable=False)

    # The common query is "everything from node X between t0 and t1", so index
    # those two columns together as well as individually.
    __table_args__ = (Index("ix_readings_node_timestamp", "node_id", "timestamp"),)

    @property
    def accel(self) -> list[list[float]]:
        """Return the accelerometer burst decoded from JSON.

        Returns:
            A list of ``[ax, ay, az]`` samples.
        """
        return json.loads(self.accel_json)

    @property
    def mag(self) -> list[float]:
        """Return the magnetometer vector as ``[mx, my, mz]``."""
        return [self.mag_x, self.mag_y, self.mag_z]

    def to_dict(self) -> dict[str, Any]:
        """Return the reading in the same shape as the incoming ESP32 payload.

        Downstream phases (feature engineering, the pipeline runner) consume
        either synthetic payloads or database rows.  Emitting the same shape
        from both is what lets the source be swapped with no code changes.

        Returns:
            A dictionary matching the ESP32 JSON payload, plus ``id`` and
            ``received_at`` which only exist server-side.
        """
        return {
            "id": self.id,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat() + "Z",
            "soil_raw": self.soil_raw,
            "accel": self.accel,
            "mag": self.mag,
            "temp_c": self.temp_c,
            "humidity_pct": self.humidity_pct,
            "pressure_hpa": self.pressure_hpa,
            "received_at": self.received_at.isoformat() + "Z",
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"Reading(id={self.id}, node_id={self.node_id!r}, "
            f"timestamp={self.timestamp!r})"
        )
