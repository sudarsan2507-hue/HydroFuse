"""Simulated sensor dropouts.

Real transmissions fail: an I2C read times out, a packet gets truncated, a
node reboots mid-cycle. This module models that as replacing one or more
fields of an otherwise-complete reading with ``None`` before it is written
out - the same shape of failure a flaky peripheral produces on real hardware.

A record with a ``None`` field cannot satisfy :class:`backend.schemas.ReadingIn`
(its fields are not Optional), so a dropout-affected row is not expected to
pass ``POST /ingest`` - that mirrors reality, where a real backend would
likewise reject a corrupted payload. Each generated record carries a
``dropout`` boolean flag so downstream code (tests, feature engineering) can
tell clean rows from corrupted ones without re-deriving it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# node_id and timestamp are the record's identity; a real node's packet
# header survives even when a sensor read inside it fails, so these two are
# never candidates for dropout.
_NEVER_DROPPED = {"node_id", "timestamp"}


def apply_dropout(
    record: dict[str, Any],
    rng: np.random.Generator,
    probability: float,
    candidate_fields: list[str],
    max_fields: int,
) -> tuple[dict[str, Any], bool]:
    """Possibly null out one or more fields of a single reading.

    Args:
        record: A complete payload dictionary (as it would be sent to
            ``/ingest``). Not modified in place.
        rng: Seeded random generator.
        probability: Probability that this record is affected at all.
        candidate_fields: Field names eligible to be nulled. Must all exist
            in ``record``.
        max_fields: Maximum number of fields to null out at once, when
            dropout is triggered.

    Returns:
        A tuple ``(record, dropped)``: a (possibly modified) copy of the
        record, and whether dropout was applied.

    Raises:
        ValueError: If a candidate field is one of the protected identity
            fields, or does not exist in the record.
    """
    for field in candidate_fields:
        if field in _NEVER_DROPPED:
            raise ValueError(f"{field!r} is an identity field and cannot be dropped")
        if field not in record:
            raise ValueError(f"{field!r} is not a field of this record")

    result = dict(record)
    if rng.uniform() >= probability:
        return result, False

    n_fields = int(rng.integers(1, max_fields + 1))
    n_fields = min(n_fields, len(candidate_fields))
    chosen = rng.choice(candidate_fields, size=n_fields, replace=False)
    for field in chosen:
        result[str(field)] = None

    return result, True
