"""Saving and loading synthetic scenario data as parquet.

The accelerometer and magnetometer columns hold Python lists (and,
occasionally, ``None`` from a simulated dropout). Parquet has no native type
for "list of list of float, sometimes null", so those two columns are
JSON-encoded to plain strings before writing - the same representation
:class:`backend.models.Reading` already uses for ``accel_json`` in the
database - and decoded back to Python lists on load. Every other column
round-trips through parquet with its native type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Columns that are JSON-encoded for storage because parquet cannot represent
# a nested list column that sometimes holds None.
_JSON_COLUMNS = ["accel", "mag"]

ALL_SCENARIOS_FILENAME = "all_scenarios.parquet"


def _is_missing(value: Any) -> bool:
    """Return True for a missing value, however parquet round-tripped it.

    A dropout-affected cell is written as Python ``None``, but a scalar
    round trip through pyarrow can hand it back as ``None``, ``float('nan')``
    or ``pandas.NA`` depending on the column's inferred dtype. A list-valued
    cell (accel/mag) is never itself NaN, so ``pd.isna`` is only safe to call
    on something that is not a list.

    Args:
        value: The cell value to check.

    Returns:
        True if the value represents "missing", False otherwise.
    """
    if isinstance(value, list):
        return False
    return bool(pd.isna(value))


def _encode_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the list-valued columns JSON-encoded.

    Args:
        df: A scenario DataFrame as produced by
            :func:`synthetic.generator.generate_scenario`.

    Returns:
        A copy where each column in ``_JSON_COLUMNS`` holds a JSON string
        (or ``None``) instead of a Python list.
    """
    encoded = df.copy()
    for column in _JSON_COLUMNS:
        if column in encoded.columns:
            encoded[column] = encoded[column].apply(lambda v: None if _is_missing(v) else json.dumps(v))
    return encoded


def _decode_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the JSON-encoded columns decoded back to lists.

    Args:
        df: A DataFrame just read back from parquet.

    Returns:
        A copy where each column in ``_JSON_COLUMNS`` holds a Python list (or
        ``None``) instead of a JSON string.
    """
    decoded = df.copy()
    for column in _JSON_COLUMNS:
        if column in decoded.columns:
            decoded[column] = decoded[column].apply(lambda v: None if _is_missing(v) else json.loads(v))
    return decoded


def save_scenario(df: pd.DataFrame, path: Path | str) -> None:
    """Write one scenario's DataFrame to a parquet file.

    Args:
        df: The scenario DataFrame to save.
        path: Destination file path. Parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _encode_json_columns(df).to_parquet(path, engine="pyarrow", index=False)


def load_scenario(path: Path | str) -> pd.DataFrame:
    """Read one scenario's parquet file back into a DataFrame.

    Args:
        path: Path to a file previously written by :func:`save_scenario`.

    Returns:
        The scenario DataFrame, with ``accel`` and ``mag`` decoded back to
        Python lists (or ``None`` for a dropout-affected row).
    """
    # dtype_backend="numpy_nullable" keeps soil_raw/temp_c/humidity_pct/
    # pressure_hpa as nullable Int64/Float64 columns, so a dropout-affected
    # cell comes back as pandas.NA rather than being silently collapsed into
    # a plain NaN on a float64 column (which would also have quietly turned
    # soil_raw from an integer into a float).
    raw = pd.read_parquet(Path(path), engine="pyarrow", dtype_backend="numpy_nullable")
    return _decode_json_columns(raw)


def save_all(frames: dict[str, pd.DataFrame], output_dir: Path | str) -> dict[str, Path]:
    """Save every scenario plus a combined file, all under one directory.

    Args:
        frames: Mapping from scenario name to its DataFrame.
        output_dir: Directory to write into. Created if missing.

    Returns:
        Mapping from scenario name (plus the special key ``"all"``) to the
        path each file was written to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, df in frames.items():
        path = output_dir / f"{name}.parquet"
        save_scenario(df, path)
        written[name] = path

    combined = pd.concat(frames.values(), ignore_index=True)
    combined_path = output_dir / ALL_SCENARIOS_FILENAME
    save_scenario(combined, combined_path)
    written["all"] = combined_path

    return written


def row_to_payload(row: Any) -> dict[str, Any]:
    """Build an ESP32-shaped payload dictionary from one DataFrame row.

    This produces exactly the fields ``POST /ingest`` expects (see
    :class:`backend.schemas.ReadingIn`), dropping the label columns. It is
    meant for a clean (non-dropout) row; a dropout row will still produce a
    dict of the same shape, just with ``None`` in place of whatever field was
    dropped, which will fail validation the same way a corrupted real
    payload would.

    Args:
        row: A row from a scenario DataFrame (e.g. from ``df.itertuples()``
            or ``df.iloc[i]``).

    Returns:
        A dictionary with the seven ESP32 payload fields.
    """
    timestamp = row.timestamp
    return {
        "node_id": row.node_id,
        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        "soil_raw": _scalar_or_none(row.soil_raw, int),
        "accel": row.accel,
        "mag": row.mag,
        "temp_c": _scalar_or_none(row.temp_c, float),
        "humidity_pct": _scalar_or_none(row.humidity_pct, float),
        "pressure_hpa": _scalar_or_none(row.pressure_hpa, float),
    }


def _scalar_or_none(value: Any, cast: type) -> Any:
    """Return ``None`` for a missing scalar, otherwise ``cast(value)``.

    Guards against every representation a nullable pandas dtype might hand
    back for a dropout-affected cell (``pandas.NA``, ``numpy.nan``, ``None``),
    so the payload always carries an honest Python ``None`` - which is what
    JSON ``null`` decodes to - never a NaN that would silently pass through
    as a (wrong) floating-point value.

    Args:
        value: The cell value, possibly missing.
        cast: Type to convert a present value to (``int`` or ``float``).

    Returns:
        ``None`` if the value is missing, otherwise ``cast(value)``.
    """
    if pd.isna(value):
        return None
    return cast(value)
