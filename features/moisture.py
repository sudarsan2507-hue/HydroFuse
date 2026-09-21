"""Soil moisture features: calibration, rate of change, and rise detection.

Every function is pure and works on plain ``numpy``/``pandas`` series, not on
config or raw payload dicts, so each is independently unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hours_since_first(timestamp: pd.Series) -> pd.Series:
    """Convert a timestamp series to float hours since its first value.

    Dividing a ``Timedelta`` series (rather than casting datetimes to int64
    and guessing the storage unit) is safe regardless of whether pandas is
    storing the column at nanosecond, microsecond, or second resolution.

    Args:
        timestamp: A series of datetimes, any order.

    Returns:
        Float hours elapsed since ``timestamp``'s first value, same index.
    """
    parsed = pd.to_datetime(timestamp)
    return (parsed - parsed.iloc[0]) / pd.Timedelta(hours=1)


def calibrate_moisture_pct(soil_raw: pd.Series, raw_dry: float, raw_wet: float) -> pd.Series:
    """Convert a raw capacitive probe reading into a 0-100% moisture value.

    Linear interpolation between the two calibration points measured in air
    (``raw_dry`` = 0%) and in water (``raw_wet`` = 100%). Lower raw counts
    mean wetter soil, so this is a falling line, clipped to a valid range in
    case sensor drift pushes a reading past either calibration point.

    Args:
        soil_raw: Raw ADC readings.
        raw_dry: Raw value measured in air (0% moisture).
        raw_wet: Raw value measured in water (100% moisture).

    Returns:
        Moisture percentage, clipped to ``[0, 100]``.
    """
    pct = (raw_dry - soil_raw) / (raw_dry - raw_wet) * 100.0
    return pct.clip(lower=0.0, upper=100.0)


def moisture_derivative_pct_per_hour(moisture_pct: pd.Series, timestamp: pd.Series) -> pd.Series:
    """First derivative of moisture over time, in percentage points per hour.

    Args:
        moisture_pct: Calibrated moisture percentage, one value per reading.
        timestamp: The corresponding reading timestamps (same length, same
            order).

    Returns:
        The derivative, same length as the input; the first value is NaN
        (there is no prior sample to difference against).
    """
    hours = _hours_since_first(timestamp)
    return moisture_pct.diff() / hours.diff()


def rolling_slope_pct_per_hour(
    moisture_pct: pd.Series,
    timestamp: pd.Series,
    window_hours: float,
) -> pd.Series:
    """Trailing slope of moisture over a fixed lookback window, in %/hour.

    Implemented as a secant slope (value now minus value ``window_hours``
    ago, divided by the elapsed time) rather than a full rolling linear
    regression: at a roughly uniform sampling rate the two agree closely,
    and the secant form is far cheaper to compute over long series.

    Args:
        moisture_pct: Calibrated moisture percentage, one value per reading.
        timestamp: The corresponding reading timestamps (same length, sorted
            ascending).
        window_hours: Lookback window length, in hours.

    Returns:
        The slope, same length as the input; rows without a full window of
        history yet are NaN.
    """
    hours = _hours_since_first(timestamp)
    slopes = pd.Series(np.nan, index=moisture_pct.index, dtype=float)

    hours_values = hours.to_numpy()
    moisture_values = moisture_pct.to_numpy(dtype=float)

    # searchsorted finds, for each row, the earliest row at or after the
    # window's start bound - O(n log n) total rather than O(n^2).
    window_start = hours_values - window_hours
    start_indices = np.searchsorted(hours_values, window_start, side="left")

    for i in range(len(hours_values)):
        if hours_values[i] - hours_values[0] < window_hours:
            continue  # the series itself doesn't go back a full window yet
        j = start_indices[i]
        if j >= i or hours_values[i] - hours_values[j] <= 0:
            continue  # degenerate (e.g. duplicate timestamps)
        slopes.iloc[i] = (moisture_values[i] - moisture_values[j]) / (hours_values[i] - hours_values[j])

    return slopes


def time_since_sharp_rise_hours(
    moisture_derivative: pd.Series,
    timestamp: pd.Series,
    sharp_rise_threshold_pct_per_hour: float,
) -> pd.Series:
    """Hours elapsed since moisture last rose faster than a configured threshold.

    A "sharp rise" is a moisture derivative (in %/hour, wetter = positive)
    at or above the threshold - the signature of a rain event hitting the
    sensor. This tells the model how recently that happened, independent of
    whatever the moisture level or trend is doing right now.

    Args:
        moisture_derivative: Moisture derivative in %/hour (e.g. from
            :func:`moisture_derivative_pct_per_hour`).
        timestamp: The corresponding reading timestamps, sorted ascending.
        sharp_rise_threshold_pct_per_hour: Derivative magnitude above which a
            rise counts as "sharp".

    Returns:
        Hours since the most recent sharp rise, same length as the input.
        NaN for every row before the first sharp rise in the series (there
        is nothing to measure since).
    """
    hours = _hours_since_first(timestamp)
    is_sharp_rise = moisture_derivative >= sharp_rise_threshold_pct_per_hour

    last_event_hour = hours.where(is_sharp_rise).ffill()
    return hours - last_event_hour
