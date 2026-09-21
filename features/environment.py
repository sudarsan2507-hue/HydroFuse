"""Environmental features: humidity/pressure change and a rain-likelihood flag.

Pure functions on plain ``pandas`` series - no config, no I/O.
"""

from __future__ import annotations

import pandas as pd


def change_over_window(series: pd.Series, periods: int) -> pd.Series:
    """Change in a value compared to ``periods`` readings ago.

    Args:
        series: The reading-resolution series (e.g. humidity_pct).
        periods: How many readings back to compare against (derived from the
            configured feature window and the sensor's reading interval).

    Returns:
        ``series - series.shift(periods)``, same length; the first
        ``periods`` rows are NaN.
    """
    return series - series.shift(periods)


def rain_likelihood_flag(
    humidity_change_pct: pd.Series,
    pressure_change_hpa: pd.Series,
    humidity_jump_threshold_pct: float,
    pressure_drop_threshold_hpa: float,
) -> pd.Series:
    """Flag windows that look like a rain event, from humidity/pressure alone.

    This is a cheap heuristic pre-filter, independent of the trained
    classifier: humidity jumping up or pressure dropping, either one, by at
    least the configured threshold over the window.

    Args:
        humidity_change_pct: Humidity change over the feature window (see
            :func:`change_over_window`).
        pressure_change_hpa: Pressure change over the feature window.
        humidity_jump_threshold_pct: Minimum humidity rise to flag.
        pressure_drop_threshold_hpa: Minimum pressure drop (as a positive
            magnitude) to flag.

    Returns:
        A boolean Series, same length as the inputs. NaN inputs (not enough
        history yet) produce False, not a missing value - "not enough
        history to tell" is treated the same as "no rain signal seen".
    """
    humidity_jump = humidity_change_pct.fillna(0.0) >= humidity_jump_threshold_pct
    pressure_drop = pressure_change_hpa.fillna(0.0) <= -pressure_drop_threshold_hpa
    return humidity_jump | pressure_drop
