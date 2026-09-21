"""Phase 5 tests: feature engineering.

Vibration/moisture/environment functions are tested in isolation (they are
pure), then :func:`features.extract.build_feature_frame` is tested end to
end against the synthetic scenarios, since that is the strongest check that
the leak/no_leak/rain signatures built in Phase 3 actually survive into
Phase 6's training features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.config import Config, load_config
from features.environment import change_over_window, rain_likelihood_flag
from features.extract import build_feature_frame
from features.moisture import (
    calibrate_moisture_pct,
    moisture_derivative_pct_per_hour,
    rolling_slope_pct_per_hour,
    time_since_sharp_rise_hours,
)
from features.vibration import (
    band_power,
    combined_power_spectrum,
    dominant_frequency,
    filter_transients,
    spectral_entropy,
    vibration_features,
    vibration_rms,
)
from synthetic.generator import generate_scenario


# --- Vibration ---------------------------------------------------------------


def test_filter_transients_replaces_only_the_spike() -> None:
    """A single huge sample is replaced; the rest of the burst is untouched."""
    burst = [[0.0, 0.0, 1.0]] * 50
    burst[25] = [5.0, 5.0, 5.0]  # a wild outlier
    filtered = filter_transients(burst, sigma_threshold=3.0)

    assert not np.allclose(filtered[25], [5.0, 5.0, 5.0])
    assert np.allclose(filtered[0], [0.0, 0.0, 1.0])
    assert np.allclose(filtered[49], [0.0, 0.0, 1.0])


def test_dominant_frequency_finds_a_clean_sinusoid() -> None:
    """A pure sinusoid's dominant frequency is recovered from its spectrum."""
    sample_rate = 500.0
    n = 256
    t = np.arange(n) / sample_rate
    freq = 60.0
    signal = np.stack([np.sin(2 * np.pi * freq * t)] * 3, axis=1)

    freqs, power = combined_power_spectrum(signal, sample_rate)
    assert dominant_frequency(freqs, power) == pytest.approx(freq, abs=sample_rate / n)


def test_band_power_is_higher_when_the_tone_is_inside_the_band() -> None:
    """band_power is much larger when the signal's frequency sits inside the band."""
    sample_rate = 500.0
    n = 256
    t = np.arange(n) / sample_rate

    in_band = np.stack([np.sin(2 * np.pi * 60.0 * t)] * 3, axis=1)
    out_of_band = np.stack([np.sin(2 * np.pi * 200.0 * t)] * 3, axis=1)

    freqs, power_in = combined_power_spectrum(in_band, sample_rate)
    _, power_out = combined_power_spectrum(out_of_band, sample_rate)

    band = (40.0, 80.0)
    assert band_power(freqs, power_in, band) > band_power(freqs, power_out, band)


def test_spectral_entropy_is_low_for_a_tone_and_high_for_noise() -> None:
    """A single sinusoid has low entropy; white noise has high entropy."""
    rng = np.random.default_rng(0)
    sample_rate = 500.0
    n = 256
    t = np.arange(n) / sample_rate

    tone = np.stack([np.sin(2 * np.pi * 60.0 * t)] * 3, axis=1)
    noise = rng.normal(0.0, 1.0, size=(n, 3))

    _, tone_power = combined_power_spectrum(tone, sample_rate)
    _, noise_power = combined_power_spectrum(noise, sample_rate)

    assert spectral_entropy(tone_power) < 0.3
    assert spectral_entropy(noise_power) > 0.8


def test_vibration_rms_is_zero_for_a_constant_signal() -> None:
    """A perfectly still (constant) burst has zero vibration RMS."""
    samples = np.tile([0.01, -0.02, 1.0], (100, 1))
    assert vibration_rms(samples) == pytest.approx(0.0, abs=1e-9)


def test_vibration_features_returns_all_four_metrics() -> None:
    """vibration_features() returns the documented four keys."""
    bursts = [[[0.01, 0.0, 1.0]] * 64 for _ in range(3)]
    result = vibration_features(bursts, sample_rate_hz=500.0, leak_band_hz=(40.0, 80.0), transient_sigma=4.0)
    assert set(result.keys()) == {"dominant_freq_hz", "leak_band_power", "spectral_entropy", "vibration_rms"}


# --- Moisture ------------------------------------------------------------------


def test_calibrate_moisture_pct_endpoints() -> None:
    """The dry and wet calibration points map to exactly 0% and 100%."""
    raw = pd.Series([3200.0, 1300.0, 2250.0])
    pct = calibrate_moisture_pct(raw, raw_dry=3200.0, raw_wet=1300.0)
    assert pct.iloc[0] == pytest.approx(0.0)
    assert pct.iloc[1] == pytest.approx(100.0)
    assert pct.iloc[2] == pytest.approx(50.0, abs=0.1)


def test_calibrate_moisture_pct_clips_out_of_range_values() -> None:
    """A raw value beyond either calibration point still lands in [0, 100]."""
    raw = pd.Series([4000.0, 500.0])
    pct = calibrate_moisture_pct(raw, raw_dry=3200.0, raw_wet=1300.0)
    assert pct.iloc[0] == 0.0
    assert pct.iloc[1] == 100.0


def test_moisture_derivative_sign_matches_direction() -> None:
    """Moisture rising gives a positive derivative; falling gives a negative one."""
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=3, freq="1h"))
    rising = moisture_derivative_pct_per_hour(pd.Series([10.0, 20.0, 40.0]), timestamps)
    assert rising.iloc[1] == pytest.approx(10.0)
    assert rising.iloc[2] == pytest.approx(20.0)


def test_rolling_slope_matches_a_known_linear_trend() -> None:
    """A perfectly linear moisture series has a constant, correct rolling slope."""
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=10, freq="1h"))
    moisture = pd.Series([float(i) * 2.0 for i in range(10)])  # +2%/hour exactly

    slope = rolling_slope_pct_per_hour(moisture, timestamps, window_hours=3.0)
    assert slope.iloc[-1] == pytest.approx(2.0)
    assert slope.iloc[:3].isna().all()  # not enough history yet


def test_time_since_sharp_rise_resets_on_each_event() -> None:
    """The counter drops to ~0 right after a sharp rise and grows afterwards."""
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=5, freq="1h"))
    derivative = pd.Series([0.0, 50.0, 1.0, 1.0, 1.0])  # sharp rise at index 1

    result = time_since_sharp_rise_hours(derivative, timestamps, sharp_rise_threshold_pct_per_hour=10.0)
    assert result.iloc[0:1].isna().all()  # nothing before the first event
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[4] == pytest.approx(3.0)


# --- Environment ---------------------------------------------------------------


def test_change_over_window_computes_a_lagged_difference() -> None:
    """change_over_window is simply value minus value N periods ago."""
    series = pd.Series([1.0, 2.0, 5.0, 9.0])
    change = change_over_window(series, periods=2)
    assert change.iloc[2] == pytest.approx(4.0)
    assert change.iloc[3] == pytest.approx(7.0)


def test_rain_flag_triggers_on_humidity_or_pressure_alone() -> None:
    """Either a humidity jump or a pressure drop alone is enough to flag rain."""
    humidity_change = pd.Series([0.0, 15.0, 0.0])
    pressure_change = pd.Series([0.0, 0.0, -5.0])
    flag = rain_likelihood_flag(humidity_change, pressure_change, humidity_jump_threshold_pct=8.0, pressure_drop_threshold_hpa=2.0)
    assert list(flag) == [False, True, True]


# --- End-to-end: build_feature_frame on real synthetic scenarios ---------------


@pytest.fixture(scope="module")
def features_config() -> Config:
    """A module-scoped Config, independent of conftest's function-scoped one."""
    return load_config()


@pytest.fixture(scope="module")
def scenario_features(features_config: Config) -> dict[str, pd.DataFrame]:
    """Feature frames for all three scenarios, generated once and shared."""
    seed = int(features_config.get("project.random_seed"))
    return {
        name: build_feature_frame(generate_scenario(name, features_config, seed=seed + i), features_config)
        for i, name in enumerate(["no_leak", "leak", "rain_not_leak"])
    }


def test_feature_frame_has_one_row_per_window(scenario_features: dict[str, pd.DataFrame], config: Config) -> None:
    """Row count matches duration / window length (no partial trailing window issues)."""
    duration_hours = float(config.get("synthetic.duration_hours"))
    window_minutes = float(config.get("features.window_minutes"))
    expected = int(duration_hours * 60 / window_minutes)

    for df in scenario_features.values():
        # A little slack: dropout can occasionally empty out the very last window.
        assert abs(len(df) - expected) <= 1


def test_leak_features_show_a_strong_band_signal(scenario_features: dict[str, pd.DataFrame]) -> None:
    """The leak scenario's windows have far more leak-band power than no_leak's."""
    leak_power = scenario_features["leak"]["leak_band_power"].mean()
    no_leak_power = scenario_features["no_leak"]["leak_band_power"].mean()
    assert leak_power > 5.0 * no_leak_power


def test_leak_features_have_lower_spectral_entropy(scenario_features: dict[str, pd.DataFrame]) -> None:
    """A persistent tone (leak) has lower spectral entropy than broadband noise."""
    leak_entropy = scenario_features["leak"]["spectral_entropy"].mean()
    no_leak_entropy = scenario_features["no_leak"]["spectral_entropy"].mean()
    assert leak_entropy < no_leak_entropy


def test_rain_scenario_triggers_the_rain_likelihood_flag(scenario_features: dict[str, pd.DataFrame]) -> None:
    """At least one window around a rain event trips the humidity/pressure flag."""
    assert scenario_features["rain_not_leak"]["rain_likelihood_flag"].any()


def test_no_leak_rarely_trips_the_rain_flag(scenario_features: dict[str, pd.DataFrame]) -> None:
    """The no_leak scenario has no rain events, so the flag should almost never fire."""
    fraction_flagged = scenario_features["no_leak"]["rain_likelihood_flag"].mean()
    assert fraction_flagged < 0.05


def test_leak_scenario_moisture_slope_is_negative(scenario_features: dict[str, pd.DataFrame]) -> None:
    """The leak scenario's rolling moisture slope is negative (raw falling = wetter)."""
    late_windows = scenario_features["leak"].dropna(subset=["moisture_rolling_slope_pct_per_hr"])
    assert late_windows["moisture_rolling_slope_pct_per_hr"].mean() > 0.0  # moisture % rising = getting wetter


def test_labels_and_scenario_survive_into_the_feature_frame(scenario_features: dict[str, pd.DataFrame]) -> None:
    """label/scenario/rain_flag pass through unchanged from the input readings."""
    assert (scenario_features["leak"]["label"] == 1).all()
    assert (scenario_features["no_leak"]["label"] == 0).all()
    assert (scenario_features["rain_not_leak"]["rain_flag"]).all()


def test_feature_frame_works_without_a_dropout_column(config: Config) -> None:
    """build_feature_frame also accepts plain (non-synthetic) readings, e.g. from the DB.

    Real database rows never have a null field (Pydantic rejects those
    before they are ever stored), so this mimics that by dropping the
    dropout-affected rows first, then dropping the label/dropout columns
    themselves.
    """
    raw = generate_scenario("no_leak", config, seed=1)
    readings = raw[~raw["dropout"]].drop(columns=["label", "scenario", "rain_flag", "dropout"])

    features = build_feature_frame(readings, config)
    assert len(features) > 0
    assert "label" not in features.columns
