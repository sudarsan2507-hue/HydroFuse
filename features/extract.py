"""Windowed feature extraction: raw readings in, one feature row per window out.

This is the only module in :mod:`features` that reads
:class:`~backend.config.Config`; every feature function it calls
(:mod:`features.vibration`, :mod:`features.moisture`,
:mod:`features.environment`) is pure. It accepts a readings DataFrame shaped
like the ESP32 payload (the same shape whether it came from the synthetic
generator or the database - see Phase 9's ``--source`` switch), so this
function never needs to know where its input came from.
"""

from __future__ import annotations

import pandas as pd

from backend.config import Config
from features.environment import change_over_window, rain_likelihood_flag
from features.moisture import (
    calibrate_moisture_pct,
    moisture_derivative_pct_per_hour,
    rolling_slope_pct_per_hour,
    time_since_sharp_rise_hours,
)
from features.vibration import vibration_features
from satellite.pipeline import align_to_sensor_timestamps

# Label/metadata columns carried through from the input (if present) but
# never computed from the sensor values themselves.
_PASSTHROUGH_COLUMNS = ["label", "scenario", "rain_flag"]


def _reading_periods_per_window(config: Config) -> int:
    """Number of readings that make up one feature window.

    Args:
        config: Project configuration.

    Returns:
        ``window_minutes * 60 / reading_interval_s``, rounded, at least 1.
    """
    window_s = float(config.get("features.window_minutes")) * 60.0
    interval_s = float(config.get("sensor.reading_interval_s"))
    return max(1, round(window_s / interval_s))


def _add_per_reading_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute every reading-resolution derived column (moisture, environment).

    Args:
        df: One node's readings, sorted by timestamp, dropout rows already
            removed.
        config: Project configuration.

    Returns:
        A copy of ``df`` with moisture/environment feature columns added.
    """
    df = df.copy()

    df["moisture_pct"] = calibrate_moisture_pct(
        df["soil_raw"].astype(float),
        raw_dry=float(config.get("sensor.soil_raw_dry")),
        raw_wet=float(config.get("sensor.soil_raw_wet")),
    )
    df["moisture_deriv_pct_per_hr"] = moisture_derivative_pct_per_hour(df["moisture_pct"], df["timestamp"])
    df["moisture_rolling_slope_pct_per_hr"] = rolling_slope_pct_per_hour(
        df["moisture_pct"], df["timestamp"], window_hours=float(config.get("features.moisture_slope_hours"))
    )
    df["time_since_sharp_rise_hr"] = time_since_sharp_rise_hours(
        df["moisture_deriv_pct_per_hr"],
        df["timestamp"],
        sharp_rise_threshold_pct_per_hour=float(config.get("features.sharp_rise_pct_per_hour")),
    )

    periods = _reading_periods_per_window(config)
    df["humidity_change_pct"] = change_over_window(df["humidity_pct"].astype(float), periods)
    df["pressure_change_hpa"] = change_over_window(df["pressure_hpa"].astype(float), periods)
    df["rain_likelihood_flag"] = rain_likelihood_flag(
        df["humidity_change_pct"],
        df["pressure_change_hpa"],
        humidity_jump_threshold_pct=float(config.get("features.rain.humidity_jump_pct")),
        pressure_drop_threshold_hpa=float(config.get("features.rain.pressure_drop_hpa")),
    )
    return df


def _windowed_rows(node_df: pd.DataFrame, config: Config) -> list[dict]:
    """Group one node's per-reading rows into feature windows.

    Args:
        node_df: One node's readings with per-reading columns already added,
            sorted by timestamp.
        config: Project configuration.

    Returns:
        One dict per window, with vibration features aggregated across every
        burst in the window and moisture/environment features taken as of
        the window's last reading.
    """
    window_minutes = float(config.get("features.window_minutes"))
    sample_rate_hz = float(config.get("sensor.accel_sample_rate_hz"))
    leak_band = tuple(float(v) for v in config.get("synthetic.leak_band_hz"))
    transient_sigma = float(config.get("features.transient_sigma"))

    start_time = node_df["timestamp"].iloc[0]
    window_index = ((node_df["timestamp"] - start_time).dt.total_seconds() // (window_minutes * 60.0)).astype(int)

    rows: list[dict] = []
    for _window_id, group in node_df.groupby(window_index):
        last = group.iloc[-1]
        bursts = [b for b in group["accel"] if b is not None]
        if not bursts:
            continue  # every reading in this window was dropped; nothing to extract

        row = {
            "node_id": last["node_id"],
            "window_end": last["timestamp"],
            "n_readings": len(group),
            "moisture_pct": last["moisture_pct"],
            "moisture_deriv_pct_per_hr": last["moisture_deriv_pct_per_hr"],
            "moisture_rolling_slope_pct_per_hr": last["moisture_rolling_slope_pct_per_hr"],
            "time_since_sharp_rise_hr": last["time_since_sharp_rise_hr"],
            "humidity_change_pct": last["humidity_change_pct"],
            "pressure_change_hpa": last["pressure_change_hpa"],
            "rain_likelihood_flag": bool(last["rain_likelihood_flag"]),
        }
        row.update(vibration_features(bursts, sample_rate_hz, leak_band, transient_sigma))

        for column in _PASSTHROUGH_COLUMNS:
            if column in group.columns:
                row[column] = last[column]

        rows.append(row)

    return rows


def build_feature_frame(
    readings: pd.DataFrame,
    config: Config,
    satellite_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn raw sensor readings into one feature row per (node, time window).

    Args:
        readings: Readings shaped like the ESP32 payload - columns
            ``node_id``, ``timestamp``, ``soil_raw``, ``accel``, ``mag``,
            ``temp_c``, ``humidity_pct``, ``pressure_hpa`` - plus, optionally,
            ``label``/``scenario``/``rain_flag``/``dropout`` from the
            synthetic generator. Works identically whether it came from
            :mod:`synthetic.generator` or the database.
        config: Project configuration.
        satellite_df: Optional satellite features indexed by date (as
            returned by :func:`satellite.pipeline.get_satellite_dataframe`).
            If given, every satellite column plus ``satellite_age_days`` is
            joined onto each window using its end timestamp.

    Returns:
        A tidy DataFrame, one row per feature window, sorted by node then
        time. Windows where every reading was dropped are skipped rather
        than filled with fabricated values.
    """
    df = readings.copy()
    timestamps = pd.to_datetime(df["timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    df["timestamp"] = timestamps

    if "dropout" in df.columns:
        df = df[~df["dropout"]]

    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    all_rows: list[dict] = []
    for _node_id, node_df in df.groupby("node_id", sort=False):
        node_df = _add_per_reading_columns(node_df.reset_index(drop=True), config)
        all_rows.extend(_windowed_rows(node_df, config))

    features = pd.DataFrame(all_rows).sort_values(["node_id", "window_end"]).reset_index(drop=True)

    if satellite_df is not None and not features.empty:
        aligned = align_to_sensor_timestamps(satellite_df, features["window_end"])
        aligned = aligned.rename(columns={"timestamp": "window_end"})
        features = features.merge(aligned, on="window_end", how="left")

    return features
