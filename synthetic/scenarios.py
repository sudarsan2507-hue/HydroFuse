"""Pure physical models for the three labelled scenarios.

Every function here takes plain numbers/arrays and an explicit
``numpy.random.Generator`` - never a global random state and never a
:class:`~backend.config.Config` object - so each one is independently
testable and reproducible: the same generator state in always produces the
same numbers out.

Time is tracked as "hours since the scenario started" (a float array
``t_hours``) for the slow environmental signals, and as "seconds since the
scenario started" for the accelerometer, since vibration needs much finer
resolution than a hint of hours.

Scenario summary (see the Phase 3 request for the full behavioural spec):

* ``no_leak``      - stable soil moisture, normal daily temp/humidity/pressure
                     cycle, background vibration noise only.
* ``leak``         - soil moisture drifts steadily wetter; a continuous
                     sinusoid at the leak band's centre frequency is added to
                     every accelerometer burst; environment is normal (that is
                     what tells a leak apart from rain).
* ``rain_not_leak`` - one or more sharp moisture spikes with slow decay,
                     paired with a humidity jump, a pressure dip that
                     precedes the spike, and a small temperature dip; no
                     persistent vibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- Daily environmental cycle ---------------------------------------------


def daily_cycle(t_hours: np.ndarray, peak_hour: float, amplitude: float) -> np.ndarray:
    """Return a 24-hour sinusoid that peaks at ``peak_hour``.

    Args:
        t_hours: Time since the scenario started, in hours (may exceed 24).
        peak_hour: Local hour of day (0-24) at which the cycle peaks.
        amplitude: Peak deviation from zero.

    Returns:
        ``amplitude * cos(...)`` evaluated at each time, peaking whenever
        ``t_hours`` is at ``peak_hour`` modulo 24.
    """
    phase = 2.0 * np.pi * (t_hours - peak_hour) / 24.0
    return amplitude * np.cos(phase)


# --- Soil moisture (raw ADC counts; lower = wetter) -------------------------


def soil_no_leak(
    t_hours: np.ndarray,
    rng: np.random.Generator,
    baseline_raw: float,
    daily_amplitude_raw: float,
    noise_sigma_raw: float,
    peak_hour: float,
) -> np.ndarray:
    """Stable soil moisture with a slow daily drift and gaussian noise.

    Warmer afternoons dry the topsoil a little, so the raw count is modelled
    as peaking (driest) at ``peak_hour``, same as temperature.

    Args:
        t_hours: Time since the scenario started, in hours.
        rng: Seeded random generator.
        baseline_raw: Centre of the raw ADC reading.
        daily_amplitude_raw: +/- swing of the daily cycle.
        noise_sigma_raw: Standard deviation of the added gaussian noise.
        peak_hour: Local hour at which soil is driest.

    Returns:
        Raw ADC values, one per element of ``t_hours``.
    """
    drift = daily_cycle(t_hours, peak_hour, daily_amplitude_raw)
    noise = rng.normal(0.0, noise_sigma_raw, size=t_hours.shape)
    return baseline_raw + drift + noise


def soil_leak(
    t_hours: np.ndarray,
    rng: np.random.Generator,
    baseline_raw: float,
    drift_raw_per_hour: float,
    noise_sigma_raw: float,
) -> np.ndarray:
    """Soil moisture that gets steadily wetter (raw count falls) over time.

    A leak is modelled as a slow, steady linear trend rather than the daily
    cycle used elsewhere: the point of this scenario is a clean, persistent
    drift that a rolling-slope feature can pick out.

    Args:
        t_hours: Time since the scenario started, in hours.
        rng: Seeded random generator.
        baseline_raw: Starting raw ADC reading.
        drift_raw_per_hour: How many raw counts are lost (soil gets wetter)
            per hour. Positive values mean the reading falls over time.
        noise_sigma_raw: Standard deviation of the added gaussian noise.

    Returns:
        Raw ADC values, one per element of ``t_hours``.
    """
    trend = baseline_raw - drift_raw_per_hour * t_hours
    noise = rng.normal(0.0, noise_sigma_raw, size=t_hours.shape)
    return trend + noise


@dataclass(frozen=True)
class RainEvent:
    """One rain event's timing, used to shape the moisture/humidity/pressure/temp signals.

    Attributes:
        onset_hour: Hour (since scenario start) the moisture spike hits.
        spike_amplitude_raw: Depth of the moisture spike, in raw ADC counts.
        humidity_jump_pct: Size of the humidity jump, in percentage points.
        pressure_dip_hpa: Depth of the pressure dip, in hPa.
        temp_drop_c: Size of the temperature dip, in degrees C.
    """

    onset_hour: float
    spike_amplitude_raw: float
    humidity_jump_pct: float
    pressure_dip_hpa: float
    temp_drop_c: float


def sample_rain_events(
    rng: np.random.Generator,
    duration_hours: float,
    events_per_run: tuple[int, int],
    min_event_gap_hours: float,
    spike_amplitude_range_raw: tuple[float, float],
    humidity_jump_range_pct: tuple[float, float],
    pressure_dip_range_hpa: tuple[float, float],
    temp_drop_range_c: tuple[float, float],
    edge_margin_hours: float,
) -> list[RainEvent]:
    """Sample one or more non-overlapping rain events within the run.

    Event onset times are drawn uniformly and rejected if they land too close
    to the edges of the run or to an already-accepted event, so every event's
    full spike-and-decay shape is visible in the output.

    Args:
        rng: Seeded random generator.
        duration_hours: Length of the whole run, in hours.
        events_per_run: Inclusive ``(min, max)`` number of events to place.
        min_event_gap_hours: Minimum spacing between two events' onsets.
        spike_amplitude_range_raw: Range to sample each event's moisture
            spike depth from.
        humidity_jump_range_pct: Range to sample each event's humidity jump.
        pressure_dip_range_hpa: Range to sample each event's pressure dip.
        temp_drop_range_c: Range to sample each event's temperature dip.
        edge_margin_hours: Minimum distance an onset must keep from the start
            and end of the run.

    Returns:
        A list of :class:`RainEvent`, sorted by onset time.
    """
    n_events = int(rng.integers(events_per_run[0], events_per_run[1] + 1))
    lo = edge_margin_hours
    hi = max(duration_hours - edge_margin_hours, lo + 0.001)

    onsets: list[float] = []
    max_attempts = 200
    for _ in range(n_events):
        for _attempt in range(max_attempts):
            candidate = float(rng.uniform(lo, hi))
            if all(abs(candidate - existing) >= min_event_gap_hours for existing in onsets):
                onsets.append(candidate)
                break
        # If max_attempts is exhausted (a very short run with many events
        # requested), we simply place fewer events than asked rather than
        # raising - a stable, always-producible output beats a crash.
    onsets.sort()

    events = [
        RainEvent(
            onset_hour=onset,
            spike_amplitude_raw=float(rng.uniform(*spike_amplitude_range_raw)),
            humidity_jump_pct=float(rng.uniform(*humidity_jump_range_pct)),
            pressure_dip_hpa=float(rng.uniform(*pressure_dip_range_hpa)),
            temp_drop_c=float(rng.uniform(*temp_drop_range_c)),
        )
        for onset in onsets
    ]
    return events


def soil_rain(
    t_hours: np.ndarray,
    rng: np.random.Generator,
    baseline_raw: float,
    daily_amplitude_raw: float,
    noise_sigma_raw: float,
    peak_hour: float,
    events: list[RainEvent],
    decay_time_constant_hours: float,
) -> np.ndarray:
    """Soil moisture with the normal daily cycle plus sharp rain spikes.

    Each event instantly drops the raw reading (gets wetter) at its onset,
    then relaxes back toward baseline with an exponential decay - "sharp
    spike then slow decay", as opposed to the leak's steady linear drift.

    Args:
        t_hours: Time since the scenario started, in hours.
        rng: Seeded random generator.
        baseline_raw: Centre of the raw ADC reading absent any rain.
        daily_amplitude_raw: +/- swing of the normal daily cycle.
        noise_sigma_raw: Standard deviation of the added gaussian noise.
        peak_hour: Local hour at which soil is normally driest.
        events: Rain events to superimpose.
        decay_time_constant_hours: Exponential decay time constant (tau).

    Returns:
        Raw ADC values, one per element of ``t_hours``.
    """
    background = soil_no_leak(t_hours, rng, baseline_raw, daily_amplitude_raw, noise_sigma_raw, peak_hour)

    spikes = np.zeros_like(t_hours)
    for event in events:
        delta = t_hours - event.onset_hour
        active = delta >= 0.0
        spikes[active] -= event.spike_amplitude_raw * np.exp(-delta[active] / decay_time_constant_hours)

    return background + spikes


# --- Environment: temperature, humidity, pressure --------------------------


def environment_normal(
    t_hours: np.ndarray,
    rng: np.random.Generator,
    temp_baseline_c: float,
    temp_daily_amplitude_c: float,
    temp_noise_sigma_c: float,
    humidity_baseline_pct: float,
    humidity_daily_amplitude_pct: float,
    humidity_noise_sigma_pct: float,
    pressure_baseline_hpa: float,
    pressure_drift_step_hpa: float,
    pressure_noise_sigma_hpa: float,
    peak_hour: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal daily temperature/humidity cycle and slowly drifting pressure.

    Used by both ``no_leak`` and ``leak``: a leak's signature is purely
    mechanical (moisture + vibration), so its environment must look exactly
    like a normal day.

    Temperature peaks at ``peak_hour``; humidity runs in antiphase to
    temperature (dips when it's hottest). Pressure follows a bounded random
    walk rather than a cycle, since synoptic pressure changes are not tied to
    the time of day.

    Args:
        t_hours: Time since the scenario started, in hours.
        rng: Seeded random generator.
        temp_baseline_c: Mean air temperature.
        temp_daily_amplitude_c: +/- daily swing of temperature.
        temp_noise_sigma_c: Gaussian noise added to temperature.
        humidity_baseline_pct: Mean relative humidity.
        humidity_daily_amplitude_pct: +/- daily swing of humidity.
        humidity_noise_sigma_pct: Gaussian noise added to humidity.
        pressure_baseline_hpa: Starting/centring pressure.
        pressure_drift_step_hpa: Standard deviation of each step of the
            pressure random walk.
        pressure_noise_sigma_hpa: Gaussian measurement noise on pressure.
        peak_hour: Local hour at which temperature peaks.

    Returns:
        A tuple ``(temp_c, humidity_pct, pressure_hpa)``, each the same shape
        as ``t_hours``.
    """
    temp = temp_baseline_c + daily_cycle(t_hours, peak_hour, temp_daily_amplitude_c)
    temp += rng.normal(0.0, temp_noise_sigma_c, size=t_hours.shape)

    # Humidity peaks at night (antiphase to temperature): peak_hour + 12.
    humidity = humidity_baseline_pct + daily_cycle(t_hours, peak_hour + 12.0, humidity_daily_amplitude_pct)
    humidity += rng.normal(0.0, humidity_noise_sigma_pct, size=t_hours.shape)
    humidity = np.clip(humidity, 0.0, 100.0)

    pressure_walk = np.cumsum(rng.normal(0.0, pressure_drift_step_hpa, size=t_hours.shape))
    pressure = pressure_baseline_hpa + pressure_walk
    pressure += rng.normal(0.0, pressure_noise_sigma_hpa, size=t_hours.shape)

    return temp, humidity, pressure


def environment_rain(
    t_hours: np.ndarray,
    rng: np.random.Generator,
    events: list[RainEvent],
    humidity_decay_time_constant_hours: float,
    pressure_lead_time_hours: float,
    pressure_recovery_time_constant_hours: float,
    temp_recovery_time_constant_hours: float,
    **normal_kwargs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal environmental cycle with rain events overlaid.

    * Humidity jumps instantly at each event's onset, then decays back down.
    * Pressure starts dipping ``pressure_lead_time_hours`` before the onset
      (a rain-bearing system typically shows falling pressure first), bottoms
      out at the onset, then recovers.
    * Temperature dips slightly starting at the onset, then recovers.

    Args:
        t_hours: Time since the scenario started, in hours.
        rng: Seeded random generator.
        events: Rain events to overlay.
        humidity_decay_time_constant_hours: Humidity jump decay tau.
        pressure_lead_time_hours: How long before onset the pressure dip
            starts ramping in.
        pressure_recovery_time_constant_hours: Pressure recovery tau after
            onset.
        temp_recovery_time_constant_hours: Temperature recovery tau after
            onset.
        **normal_kwargs: Forwarded to :func:`environment_normal` for the
            background cycle (baselines, amplitudes, noise sigmas, peak_hour).

    Returns:
        A tuple ``(temp_c, humidity_pct, pressure_hpa)``.
    """
    temp, humidity, pressure = environment_normal(t_hours, rng, **normal_kwargs)

    for event in events:
        delta = t_hours - event.onset_hour  # hours relative to onset

        # Humidity: jumps at onset, decays afterwards.
        after = delta >= 0.0
        humidity_bump = np.zeros_like(t_hours)
        humidity_bump[after] = event.humidity_jump_pct * np.exp(-delta[after] / humidity_decay_time_constant_hours)
        humidity = humidity + humidity_bump

        # Pressure: ramps down to a trough at onset, then recovers.
        pressure_dip = np.zeros_like(t_hours)
        ramping = (delta >= -pressure_lead_time_hours) & (delta < 0.0)
        pressure_dip[ramping] = event.pressure_dip_hpa * (1.0 - (-delta[ramping] / pressure_lead_time_hours))
        pressure_dip[after] = event.pressure_dip_hpa * np.exp(-delta[after] / pressure_recovery_time_constant_hours)
        pressure = pressure - pressure_dip

        # Temperature: dips at onset, recovers afterwards.
        temp_dip = np.zeros_like(t_hours)
        temp_dip[after] = event.temp_drop_c * np.exp(-delta[after] / temp_recovery_time_constant_hours)
        temp = temp - temp_dip

    humidity = np.clip(humidity, 0.0, 100.0)
    return temp, humidity, pressure


# --- Magnetometer ------------------------------------------------------------


def magnetometer_walk(
    n_readings: int,
    rng: np.random.Generator,
    baseline_ut: tuple[float, float, float],
    step_sigma_ut: float,
) -> np.ndarray:
    """A slow random walk around a baseline vector, shared by every scenario.

    Args:
        n_readings: Number of readings to generate one magnetometer vector for.
        rng: Seeded random generator.
        baseline_ut: Centre of the walk, ``(mx, my, mz)`` in microtesla.
        step_sigma_ut: Standard deviation of each step.

    Returns:
        An ``(n_readings, 3)`` array of magnetometer vectors.
    """
    steps = rng.normal(0.0, step_sigma_ut, size=(n_readings, 3))
    walk = np.cumsum(steps, axis=0)
    return np.asarray(baseline_ut, dtype=np.float64) + walk


# --- Accelerometer bursts ----------------------------------------------------


def _add_transient_spike(
    burst: np.ndarray,
    rng: np.random.Generator,
    duration_range: tuple[int, int],
    amplitude_factor_range: tuple[float, float],
    noise_sigma_g: float,
) -> np.ndarray:
    """Overlay one footstep-like impulse onto a burst, in place semantics.

    The impulse is a half-sine envelope (smooth start and end, so it does not
    itself inject a sharp discontinuity) applied to all three axes with
    independent random sign and a small per-axis amplitude variation.

    Args:
        burst: Burst array of shape ``(n_samples, 3)``; not modified.
        rng: Seeded random generator.
        duration_range: Inclusive ``(min, max)`` spike duration, in samples.
        amplitude_factor_range: Inclusive range for the spike amplitude, as a
            multiple of ``noise_sigma_g``.
        noise_sigma_g: Baseline accelerometer noise standard deviation.

    Returns:
        A new array with the spike added.
    """
    n_samples = burst.shape[0]
    duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
    duration = min(duration, n_samples)
    start = int(rng.integers(0, max(n_samples - duration, 0) + 1))

    amplitude = float(rng.uniform(*amplitude_factor_range)) * noise_sigma_g
    envelope = np.sin(np.pi * np.arange(duration) / max(duration - 1, 1))

    result = burst.copy()
    for axis in range(3):
        axis_sign = rng.choice([-1.0, 1.0])
        axis_scale = rng.uniform(0.6, 1.0)  # footstep doesn't hit every axis equally
        result[start : start + duration, axis] += axis_sign * axis_scale * amplitude * envelope
    return result


def accel_burst(
    n_samples: int,
    sample_rate_hz: float,
    rng: np.random.Generator,
    noise_sigma_g: float,
    gravity_g: float,
    transient_probability: float,
    transient_duration_range: tuple[int, int],
    transient_amplitude_factor_range: tuple[float, float],
    leak_amplitude_g: float = 0.0,
    leak_frequency_hz: float = 0.0,
    burst_start_time_s: float = 0.0,
) -> np.ndarray:
    """Build one accelerometer burst: noise, gravity, and optional overlays.

    Args:
        n_samples: Number of samples in the burst.
        sample_rate_hz: Sampling rate.
        rng: Seeded random generator.
        noise_sigma_g: Standard deviation of the background gaussian noise.
        gravity_g: Static gravity offset applied to the Z axis.
        transient_probability: Probability this burst contains a footstep
            transient.
        transient_duration_range: Inclusive ``(min, max)`` spike duration, in
            samples.
        transient_amplitude_factor_range: Inclusive range for spike
            amplitude, as a multiple of ``noise_sigma_g``.
        leak_amplitude_g: Amplitude of the continuous leak sinusoid. 0
            disables it (used for the non-leak scenarios).
        leak_frequency_hz: Frequency of the leak sinusoid, in Hz.
        burst_start_time_s: Absolute time (seconds since the scenario
            started) of this burst's first sample. The leak sinusoid's phase
            is computed from this absolute time - not reset to zero at the
            start of every burst - so it stays coherent across the whole
            recording, matching a real, continuously vibrating pipe.

    Returns:
        An ``(n_samples, 3)`` array of ``[ax, ay, az]`` values in g.
    """
    burst = rng.normal(0.0, noise_sigma_g, size=(n_samples, 3))
    burst[:, 2] += gravity_g

    if leak_amplitude_g > 0.0:
        sample_times_s = burst_start_time_s + np.arange(n_samples) / sample_rate_hz
        leak_signal = leak_amplitude_g * np.sin(2.0 * np.pi * leak_frequency_hz * sample_times_s)
        # Applied to all three axes: a leak's vibration couples into the
        # sensor however it happens to be mounted, so we do not privilege one
        # axis over the others.
        burst[:, 0] += leak_signal
        burst[:, 1] += leak_signal
        burst[:, 2] += leak_signal

    if rng.uniform() < transient_probability:
        burst = _add_transient_spike(
            burst, rng, transient_duration_range, transient_amplitude_factor_range, noise_sigma_g
        )

    return burst
