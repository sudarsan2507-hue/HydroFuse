"""Top-level satellite pipeline: fetch (or fall back), save, and align to sensor time.

:func:`get_satellite_dataframe` is the only function most callers need: it
tries Google Earth Engine and transparently falls back to a small synthetic
frame if GEE is not authenticated/configured, so the rest of the pipeline
never has to know which source it got. :func:`align_to_sensor_timestamps` is
the bridge between satellite time (one row per day, with gaps) and sensor
time (one row every minute): it forward-fills the most recent satellite
observation onto each sensor timestamp and records how stale that
observation was, in an explicit ``satellite_age_days`` column, since a
5-day-old SMAP reading should not be trusted the same as a same-day one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import Config
from satellite.gee_client import SatelliteUnavailable, fetch_satellite_dataframe
from satellite.synthetic_fallback import SATELLITE_COLUMNS, generate_synthetic_satellite_frame


def get_satellite_dataframe(config: Config, seed: int | None = None) -> pd.DataFrame:
    """Fetch satellite features for the configured area/date range.

    Tries Google Earth Engine first; if it is not configured, not yet
    authenticated, or unreachable, falls back to a small synthetic frame with
    the same columns, so the rest of the pipeline runs unmodified either way.

    Args:
        config: Project configuration.
        seed: Random seed for the synthetic fallback (only used if GEE is
            unavailable). Defaults to ``project.random_seed``.

    Returns:
        A DataFrame indexed by date, one row per day, columns matching
        :data:`satellite.synthetic_fallback.SATELLITE_COLUMNS`.
    """
    try:
        df = fetch_satellite_dataframe(config)
        if df.empty:
            raise SatelliteUnavailable("Earth Engine returned no data for this area/date range")
        return df
    except SatelliteUnavailable as exc:
        print(f"[satellite] Earth Engine unavailable ({exc}); using synthetic fallback data.")
        return generate_synthetic_satellite_frame(config, seed=seed)


def save_satellite_dataframe(df: pd.DataFrame, config: Config) -> Path:
    """Save a satellite DataFrame to the configured parquet path.

    Args:
        df: A satellite DataFrame, indexed by date.
        config: Project configuration (``satellite.output_path``).

    Returns:
        The absolute path the file was written to.
    """
    path = config.resolve_path(config.get("satellite.output_path"))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Parquet needs a named, regular column for the index to round-trip cleanly.
    df.rename_axis("date").reset_index().to_parquet(path, engine="pyarrow", index=False)
    return path


def load_satellite_dataframe(config: Config) -> pd.DataFrame:
    """Load a previously saved satellite DataFrame.

    Args:
        config: Project configuration (``satellite.output_path``).

    Returns:
        A DataFrame indexed by date.
    """
    path = config.resolve_path(config.get("satellite.output_path"))
    df = pd.read_parquet(path, engine="pyarrow")
    return df.set_index("date")


def align_to_sensor_timestamps(
    satellite_df: pd.DataFrame,
    sensor_timestamps: pd.Series,
) -> pd.DataFrame:
    """Forward-fill satellite features onto sensor timestamps, with an age column.

    Satellite revisit is days; sensor readings are minutes. For each sensor
    timestamp, this attaches the most recent satellite observation at or
    before that time (never a future one - that would leak information) and
    records how old it was in ``satellite_age_days``. A sensor timestamp
    earlier than the first available satellite date gets all-NaN satellite
    columns and a NaN age, rather than a fabricated value.

    Args:
        satellite_df: Satellite features indexed by date (one row per day,
            gaps allowed - e.g. cloud-masked days are simply absent).
        sensor_timestamps: The sensor reading timestamps to align onto, as a
            Series of datetimes (need not be sorted).

    Returns:
        A DataFrame with one row per input timestamp, the original
        ``sensor_timestamps`` as a ``timestamp`` column, every satellite
        feature column forward-filled, and ``satellite_age_days`` (float,
        NaN where no satellite data precedes the timestamp).
    """
    satellite_sorted = satellite_df.sort_index().rename_axis("satellite_date").reset_index()

    # merge_asof requires the left side sorted by the join key; track the
    # caller's original order in `_order` so it can be restored afterwards.
    sensor_df = pd.DataFrame({"timestamp": pd.to_datetime(sensor_timestamps).reset_index(drop=True)})
    sensor_df["_order"] = range(len(sensor_df))
    sensor_df = sensor_df.sort_values("timestamp")

    merged = pd.merge_asof(
        sensor_df,
        satellite_sorted,
        left_on="timestamp",
        right_on="satellite_date",
        direction="backward",
    )
    merged["satellite_age_days"] = (
        merged["timestamp"] - merged["satellite_date"]
    ).dt.total_seconds() / 86400.0
    merged = merged.drop(columns=["satellite_date"])

    return merged.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
