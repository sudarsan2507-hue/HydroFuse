"""Phase 4 tests: satellite pipeline.

Google Earth Engine cannot be exercised in CI (it needs interactive OAuth and
network access), so these tests cover the part that always runs: the
synthetic fallback frame, saving/loading, and the sensor-timestamp alignment
with its age-of-data column.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.config import Config
from satellite.pipeline import (
    align_to_sensor_timestamps,
    get_satellite_dataframe,
    load_satellite_dataframe,
    save_satellite_dataframe,
)
from satellite.synthetic_fallback import SATELLITE_COLUMNS, generate_synthetic_satellite_frame


def test_synthetic_frame_has_expected_columns_and_span(config: Config) -> None:
    """The fallback frame covers the configured date range with every column."""
    df = generate_synthetic_satellite_frame(config)
    assert list(df.columns) == SATELLITE_COLUMNS
    assert df.index.min().strftime("%Y-%m-%d") == config.get("date_range.start")
    assert df.index.max().strftime("%Y-%m-%d") == config.get("date_range.end")
    assert df.notna().all().all()


def test_synthetic_frame_is_reproducible(config: Config) -> None:
    """The same seed produces the same synthetic satellite frame."""
    first = generate_synthetic_satellite_frame(config, seed=7)
    second = generate_synthetic_satellite_frame(config, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_get_satellite_dataframe_falls_back_without_gee_project(config: Config) -> None:
    """With no gee_project_id configured, get_satellite_dataframe uses the fallback."""
    assert config.get("satellite.gee_project_id", None) in (None, "")
    df = get_satellite_dataframe(config, seed=1)
    assert list(df.columns) == SATELLITE_COLUMNS
    assert len(df) > 0


def test_save_and_load_round_trips(tmp_path, config: Config) -> None:
    """A saved satellite frame loads back with the same shape and index."""
    from backend.config import Config as ConfigClass

    patched = config.as_dict()
    patched["satellite"] = dict(patched["satellite"])
    patched["satellite"]["output_path"] = str(tmp_path / "satellite.parquet")
    patched_config = ConfigClass(patched, source_path=config.source_path)

    df = generate_synthetic_satellite_frame(config, seed=1)
    path = save_satellite_dataframe(df, patched_config)
    assert path.exists()

    loaded = load_satellite_dataframe(patched_config)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)


# --- Alignment onto sensor timestamps ---------------------------------------


@pytest.fixture()
def small_satellite_df() -> pd.DataFrame:
    """A tiny 3-day satellite frame with a gap on day 2 (e.g. cloud-masked)."""
    dates = pd.to_datetime(["2025-01-01", "2025-01-03"])
    return pd.DataFrame({"ndvi": [0.4, 0.5], "smap_soil_moisture": [0.2, 0.25]}, index=dates)


def test_alignment_forward_fills_the_most_recent_observation(small_satellite_df: pd.DataFrame) -> None:
    """A sensor timestamp gets the latest satellite row at or before it."""
    sensor_timestamps = pd.Series(pd.to_datetime(["2025-01-02 12:00:00", "2025-01-04 00:00:00"]))
    aligned = align_to_sensor_timestamps(small_satellite_df, sensor_timestamps)

    # 2025-01-02 has no direct satellite pass; it should carry Jan 1's values.
    row_jan2 = aligned[aligned["timestamp"] == "2025-01-02 12:00:00"].iloc[0]
    assert row_jan2["ndvi"] == pytest.approx(0.4)

    # 2025-01-04 should carry Jan 3's values (the most recent available).
    row_jan4 = aligned[aligned["timestamp"] == "2025-01-04 00:00:00"].iloc[0]
    assert row_jan4["ndvi"] == pytest.approx(0.5)


def test_alignment_age_grows_with_staleness(small_satellite_df: pd.DataFrame) -> None:
    """satellite_age_days reflects exactly how long ago the source observation was."""
    sensor_timestamps = pd.Series(pd.to_datetime(["2025-01-01 06:00:00", "2025-01-02 18:00:00"]))
    aligned = align_to_sensor_timestamps(small_satellite_df, sensor_timestamps).set_index("timestamp")

    assert aligned.loc[pd.Timestamp("2025-01-01 06:00:00"), "satellite_age_days"] == pytest.approx(0.25)
    # Jan 1 00:00 satellite data, sensor reading at Jan 2 18:00 -> 1.75 days old.
    assert aligned.loc[pd.Timestamp("2025-01-02 18:00:00"), "satellite_age_days"] == pytest.approx(1.75)


def test_alignment_before_first_satellite_date_is_nan(small_satellite_df: pd.DataFrame) -> None:
    """A sensor reading earlier than any satellite pass gets NaN, not a guess."""
    sensor_timestamps = pd.Series(pd.to_datetime(["2024-12-31 00:00:00"]))
    aligned = align_to_sensor_timestamps(small_satellite_df, sensor_timestamps)
    assert aligned["ndvi"].isna().all()
    assert aligned["satellite_age_days"].isna().all()


def test_alignment_never_uses_a_future_observation(small_satellite_df: pd.DataFrame) -> None:
    """Alignment is strictly backward-looking - no data leakage from the future."""
    sensor_timestamps = pd.Series(pd.to_datetime(["2025-01-01 00:00:00"]))
    aligned = align_to_sensor_timestamps(small_satellite_df, sensor_timestamps)
    # Exactly at Jan 1's own timestamp: age must be 0, not negative, and must
    # not have picked up Jan 3's later (and closer-in-value) row.
    assert aligned.iloc[0]["satellite_age_days"] == pytest.approx(0.0)
    assert aligned.iloc[0]["ndvi"] == pytest.approx(0.4)


def test_alignment_preserves_input_row_order(small_satellite_df: pd.DataFrame) -> None:
    """Output rows correspond 1:1 to input timestamps, in the original order."""
    sensor_timestamps = pd.Series(pd.to_datetime(["2025-01-04", "2025-01-01", "2025-01-02"]))
    aligned = align_to_sensor_timestamps(small_satellite_df, sensor_timestamps)
    assert list(aligned["timestamp"]) == list(pd.to_datetime(sensor_timestamps))
