"""Top-level synthetic data generation: config in, labelled DataFrame out.

This module is the only place that reads :class:`~backend.config.Config` for
synthetic data; every physical model it calls (in :mod:`synthetic.scenarios`)
is a pure function of plain numbers and a random generator. That split is
what keeps the scenario physics unit-testable in isolation from YAML parsing.

Each generated row has exactly the fields of the real ESP32 payload
(``node_id``, ``timestamp``, ``soil_raw``, ``accel``, ``mag``, ``temp_c``,
``humidity_pct``, ``pressure_hpa``) plus four label columns used for training
and analysis, never sent to a real backend: ``label`` (0/1), ``scenario``
(name), ``rain_flag`` (bool) and ``dropout`` (bool, whether this row had a
field nulled out).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.config import Config, ConfigError
from synthetic import scenarios
from synthetic.dropout import apply_dropout

SCENARIO_NO_LEAK = "no_leak"
SCENARIO_LEAK = "leak"
SCENARIO_RAIN_NOT_LEAK = "rain_not_leak"

# Payload fields, in the order the ESP32 sends them. Kept as a module
# constant so the generator and its tests agree on exactly what "the schema"
# means without repeating the list.
PAYLOAD_FIELDS = [
    "node_id",
    "timestamp",
    "soil_raw",
    "accel",
    "mag",
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
]
LABEL_FIELDS = ["label", "scenario", "rain_flag", "dropout"]

_SCENARIO_LABELS = {
    SCENARIO_NO_LEAK: 0,
    SCENARIO_LEAK: 1,
    SCENARIO_RAIN_NOT_LEAK: 0,
}


def _pair(cfg: Config, key: str) -> tuple[float, float]:
    """Read a two-element configuration list as a ``(low, high)`` tuple.

    Args:
        cfg: Project configuration.
        key: Dotted key pointing at a two-element list.

    Returns:
        The two values as floats.

    Raises:
        ConfigError: If the configured value does not have exactly two elements.
    """
    values = cfg.get(key)
    if len(values) != 2:
        raise ConfigError(f"{key} must have exactly two elements, got {values!r}")
    return float(values[0]), float(values[1])


def build_timestamps(base_time: datetime, interval_s: float, n_readings: int) -> list[datetime]:
    """Build a contiguous list of reading timestamps.

    Args:
        base_time: Timestamp of the first reading.
        interval_s: Seconds between consecutive readings.
        n_readings: Number of timestamps to produce.

    Returns:
        A list of ``n_readings`` timestamps, ``interval_s`` seconds apart,
        starting at ``base_time``.
    """
    return [base_time + timedelta(seconds=interval_s * i) for i in range(n_readings)]


def _round_burst(burst: np.ndarray, decimals: int = 5) -> list[list[float]]:
    """Convert an ``(n, 3)`` accelerometer array into a rounded list-of-lists.

    Args:
        burst: Array of shape ``(n_samples, 3)``.
        decimals: Decimal places to round each value to.

    Returns:
        A plain Python list of ``[ax, ay, az]`` lists, ready for JSON.
    """
    return np.round(burst, decimals).tolist()


def generate_scenario(scenario: str, config: Config, seed: int) -> pd.DataFrame:
    """Generate one labelled scenario as a DataFrame shaped like the ESP32 payload.

    Args:
        scenario: One of ``"no_leak"``, ``"leak"``, ``"rain_not_leak"``.
        config: Project configuration.
        seed: Random seed. The same seed always produces the same DataFrame.

    Returns:
        A DataFrame with one row per reading, columns
        :data:`PAYLOAD_FIELDS` + :data:`LABEL_FIELDS`. Dropout-affected
        fields (``soil_raw``, ``accel``, ``mag``, ``temp_c``,
        ``humidity_pct``, ``pressure_hpa``) are stored as object-dtype
        columns so a nulled value is Python ``None``, never ``NaN``.

    Raises:
        ValueError: If ``scenario`` is not a known scenario name.
    """
    if scenario not in _SCENARIO_LABELS:
        raise ValueError(f"Unknown scenario {scenario!r}, expected one of {list(_SCENARIO_LABELS)}")

    rng = np.random.default_rng(seed)

    # --- Timing ---------------------------------------------------------
    base_time = pd.Timestamp(config.get("synthetic.base_time")).to_pydatetime()
    duration_hours = float(config.get("synthetic.duration_hours"))
    interval_s = float(config.get("sensor.reading_interval_s"))
    sample_rate_hz = float(config.get("sensor.accel_sample_rate_hz"))
    burst_samples = int(config.get("sensor.accel_burst_samples"))
    node_id = str(config.get("synthetic.node_id"))

    n_readings = int(round(duration_hours * 3600.0 / interval_s))
    timestamps = build_timestamps(base_time, interval_s, n_readings)
    t_hours = np.arange(n_readings) * (interval_s / 3600.0)

    soil_min = int(config.get("sensor.soil_raw_min"))
    soil_max = int(config.get("sensor.soil_raw_max"))
    peak_hour = float(config.get("synthetic.environment.peak_hour"))

    environment_kwargs = dict(
        temp_baseline_c=float(config.get("synthetic.environment.temp_baseline_c")),
        temp_daily_amplitude_c=float(config.get("synthetic.environment.temp_daily_amplitude_c")),
        temp_noise_sigma_c=float(config.get("synthetic.environment.temp_noise_sigma_c")),
        humidity_baseline_pct=float(config.get("synthetic.environment.humidity_baseline_pct")),
        humidity_daily_amplitude_pct=float(config.get("synthetic.environment.humidity_daily_amplitude_pct")),
        humidity_noise_sigma_pct=float(config.get("synthetic.environment.humidity_noise_sigma_pct")),
        pressure_baseline_hpa=float(config.get("synthetic.environment.pressure_baseline_hpa")),
        pressure_drift_step_hpa=float(config.get("synthetic.environment.pressure_drift_step_hpa")),
        pressure_noise_sigma_hpa=float(config.get("synthetic.environment.pressure_noise_sigma_hpa")),
        peak_hour=peak_hour,
    )

    leak_band_low, leak_band_high = _pair(config, "synthetic.leak_band_hz")
    leak_frequency_hz = (leak_band_low + leak_band_high) / 2.0
    leak_amplitude_g = float(config.get("synthetic.leak_vibration_amplitude_g"))

    # --- Scenario-specific soil moisture and environment -----------------
    rain_flag = False
    if scenario == SCENARIO_NO_LEAK:
        soil_raw = scenarios.soil_no_leak(
            t_hours,
            rng,
            baseline_raw=float(config.get("synthetic.soil.no_leak_baseline_raw")),
            daily_amplitude_raw=float(config.get("synthetic.soil.no_leak_daily_amplitude_raw")),
            noise_sigma_raw=float(config.get("synthetic.soil.noise_sigma_raw")),
            peak_hour=peak_hour,
        )
        temp, humidity, pressure = scenarios.environment_normal(t_hours, rng, **environment_kwargs)
        burst_leak_amplitude = 0.0

    elif scenario == SCENARIO_LEAK:
        soil_raw = scenarios.soil_leak(
            t_hours,
            rng,
            baseline_raw=float(config.get("synthetic.soil.no_leak_baseline_raw")),
            drift_raw_per_hour=float(config.get("synthetic.soil.leak_drift_raw_per_hour")),
            noise_sigma_raw=float(config.get("synthetic.soil.noise_sigma_raw")),
        )
        # Environment stays normal: mechanical vibration + moisture drift is
        # what distinguishes a leak, not the weather.
        temp, humidity, pressure = scenarios.environment_normal(t_hours, rng, **environment_kwargs)
        burst_leak_amplitude = leak_amplitude_g

    else:  # SCENARIO_RAIN_NOT_LEAK
        rain_flag = True
        events = scenarios.sample_rain_events(
            rng,
            duration_hours=duration_hours,
            events_per_run=tuple(int(v) for v in config.get("synthetic.rain.events_per_run")),
            min_event_gap_hours=float(config.get("synthetic.rain.min_event_gap_hours")),
            spike_amplitude_range_raw=_pair(config, "synthetic.soil.rain_spike_amplitude_raw"),
            humidity_jump_range_pct=_pair(config, "synthetic.rain.humidity_jump_pct"),
            pressure_dip_range_hpa=_pair(config, "synthetic.rain.pressure_dip_hpa"),
            temp_drop_range_c=_pair(config, "synthetic.rain.temp_drop_c"),
            edge_margin_hours=float(config.get("synthetic.rain.pressure_lead_time_hours")) + 1.0,
        )
        soil_raw = scenarios.soil_rain(
            t_hours,
            rng,
            baseline_raw=float(config.get("synthetic.soil.no_leak_baseline_raw")),
            daily_amplitude_raw=float(config.get("synthetic.soil.no_leak_daily_amplitude_raw")),
            noise_sigma_raw=float(config.get("synthetic.soil.noise_sigma_raw")),
            peak_hour=peak_hour,
            events=events,
            decay_time_constant_hours=float(config.get("synthetic.soil.rain_decay_time_constant_hours")),
        )
        temp, humidity, pressure = scenarios.environment_rain(
            t_hours,
            rng,
            events=events,
            humidity_decay_time_constant_hours=float(
                config.get("synthetic.rain.humidity_decay_time_constant_hours")
            ),
            pressure_lead_time_hours=float(config.get("synthetic.rain.pressure_lead_time_hours")),
            pressure_recovery_time_constant_hours=float(
                config.get("synthetic.rain.pressure_recovery_time_constant_hours")
            ),
            temp_recovery_time_constant_hours=float(
                config.get("synthetic.rain.temp_recovery_time_constant_hours")
            ),
            **environment_kwargs,
        )
        burst_leak_amplitude = 0.0

    soil_raw = np.clip(np.round(soil_raw), soil_min, soil_max).astype(int)

    # --- Magnetometer: identical random walk in every scenario -----------
    mag_baseline = tuple(float(v) for v in config.get("synthetic.mag.baseline_ut"))
    mag = scenarios.magnetometer_walk(
        n_readings,
        rng,
        baseline_ut=mag_baseline,
        step_sigma_ut=float(config.get("synthetic.mag.random_walk_step_ut")),
    )

    # --- Accelerometer bursts ----------------------------------------------
    noise_sigma_g = float(config.get("synthetic.accel_noise_sigma_g"))
    gravity_g = float(config.get("synthetic.gravity_g"))
    transient_probability = float(config.get("synthetic.transient_spike_probability"))
    transient_duration_range = tuple(int(v) for v in config.get("synthetic.transient_spike_duration_samples"))
    transient_amplitude_range = _pair(config, "synthetic.transient_spike_amplitude_factor")

    accel_bursts = [
        _round_burst(
            scenarios.accel_burst(
                n_samples=burst_samples,
                sample_rate_hz=sample_rate_hz,
                rng=rng,
                noise_sigma_g=noise_sigma_g,
                gravity_g=gravity_g,
                transient_probability=transient_probability,
                transient_duration_range=transient_duration_range,
                transient_amplitude_factor_range=transient_amplitude_range,
                leak_amplitude_g=burst_leak_amplitude,
                leak_frequency_hz=leak_frequency_hz,
                burst_start_time_s=i * interval_s,
            )
        )
        for i in range(n_readings)
    ]

    # --- Assemble records and apply dropout -------------------------------
    dropout_probability = float(config.get("synthetic.dropout_probability"))
    dropout_fields = list(config.get("synthetic.dropout_fields"))
    dropout_max_fields = int(config.get("synthetic.dropout_max_fields"))

    records: list[dict] = []
    dropped_flags: list[bool] = []
    for i in range(n_readings):
        record = {
            "node_id": node_id,
            "timestamp": timestamps[i],
            "soil_raw": int(soil_raw[i]),
            "accel": accel_bursts[i],
            "mag": [round(float(v), 3) for v in mag[i]],
            "temp_c": round(float(temp[i]), 2),
            "humidity_pct": round(float(humidity[i]), 2),
            "pressure_hpa": round(float(pressure[i]), 2),
        }
        record, dropped = apply_dropout(record, rng, dropout_probability, dropout_fields, dropout_max_fields)
        records.append(record)
        dropped_flags.append(dropped)

    label = _SCENARIO_LABELS[scenario]

    # Built column-by-column with explicit *nullable* dtypes for any column
    # that can hold a dropout None. Plain pd.DataFrame(records) would upcast
    # a mixed int/None column to plain float64 and silently turn None into
    # NaN - a different value that breaks the dropout contract (and loses
    # soil_raw's int-ness). pandas' nullable Int64/Float64 extension dtypes
    # keep a real None (as pd.NA) alongside real integers/floats, and survive
    # a parquet round trip intact when read back with dtype_backend=
    # "numpy_nullable" (see synthetic.io.load_scenario).
    df = pd.DataFrame(
        {
            "node_id": [r["node_id"] for r in records],
            "timestamp": [r["timestamp"] for r in records],
            "soil_raw": pd.array([r["soil_raw"] for r in records], dtype="Int64"),
            "accel": pd.array([r["accel"] for r in records], dtype=object),
            "mag": pd.array([r["mag"] for r in records], dtype=object),
            "temp_c": pd.array([r["temp_c"] for r in records], dtype="Float64"),
            "humidity_pct": pd.array([r["humidity_pct"] for r in records], dtype="Float64"),
            "pressure_hpa": pd.array([r["pressure_hpa"] for r in records], dtype="Float64"),
            "label": label,
            "scenario": scenario,
            "rain_flag": rain_flag,
            "dropout": dropped_flags,
        }
    )
    return df


def generate_all(config: Config, seed: int) -> dict[str, pd.DataFrame]:
    """Generate every configured scenario.

    Each scenario gets its own derived seed (``seed`` offset by its index in
    the configured scenario list) so that changing one scenario's physics
    does not perturb another scenario's random draws, while the whole set
    remains reproducible from a single top-level seed.

    Args:
        config: Project configuration.
        seed: Top-level random seed.

    Returns:
        A mapping from scenario name to its labelled DataFrame, in the order
        given by ``synthetic.scenarios`` in the configuration.
    """
    scenario_names = list(config.get("synthetic.scenarios"))
    return {
        name: generate_scenario(name, config, seed=seed + index)
        for index, name in enumerate(scenario_names)
    }


def combine_scenarios(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate scenario frames into one combined, time-sorted-per-scenario table.

    Args:
        frames: Mapping from scenario name to its DataFrame, as returned by
            :func:`generate_all`.

    Returns:
        A single DataFrame with all rows from all scenarios, in the order the
        scenarios were given (each scenario's own rows stay in time order).
    """
    return pd.concat(frames.values(), ignore_index=True)
