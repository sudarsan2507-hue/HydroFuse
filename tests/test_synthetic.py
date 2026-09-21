"""Phase 3 tests: the synthetic scenario generator.

These tests generate real data (no mocking) and check it against the same
schema the real ESP32 payload has to satisfy, and against the physical
signatures each scenario is supposed to contain (a leak's persistent
vibration tone, a rain event's moisture spike). Frequency-domain assertions
use the *power ratio* (peak-to-mean) rather than a specific frequency value
for the no-leak/rain checks, because a pure-noise spectrum's peak bin lands
wherever chance puts it - a single strong, persistent tone is what a leak
looks like, not any particular Hz value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.config import Config, load_config
from backend.schemas import ReadingIn
from synthetic.analysis import (
    aggregate_power_spectrum,
    band_power_fraction,
    dominant_frequency,
    peak_to_mean_ratio,
)
from synthetic.generator import (
    LABEL_FIELDS,
    PAYLOAD_FIELDS,
    SCENARIO_LEAK,
    SCENARIO_NO_LEAK,
    SCENARIO_RAIN_NOT_LEAK,
    combine_scenarios,
    generate_all,
    generate_scenario,
)
from synthetic.io import load_scenario, row_to_payload, save_all, save_scenario


@pytest.fixture(scope="module")
def synthetic_config() -> Config:
    """A module-scoped copy of the project configuration.

    Generation is deterministic and read-only, so sharing one Config across
    every test in this module (rather than the function-scoped ``config``
    fixture from conftest.py) is safe and avoids re-parsing config.yaml and
    re-generating three scenarios for every single test.
    """
    return load_config()


@pytest.fixture(scope="module")
def seed(synthetic_config: Config) -> int:
    """The project's configured random seed, as an int."""
    return int(synthetic_config.get("project.random_seed"))


@pytest.fixture(scope="module")
def frames(synthetic_config: Config, seed: int) -> dict[str, pd.DataFrame]:
    """Generate all three scenarios once and share them across tests."""
    return generate_all(synthetic_config, seed=seed)


# --- Labels and flags --------------------------------------------------------


def test_no_leak_is_labelled_zero_with_no_rain_flag(frames: dict[str, pd.DataFrame]) -> None:
    """Every no_leak row has label 0, rain_flag False, and the right scenario name."""
    df = frames[SCENARIO_NO_LEAK]
    assert (df["label"] == 0).all()
    assert (~df["rain_flag"]).all()
    assert (df["scenario"] == SCENARIO_NO_LEAK).all()


def test_leak_is_labelled_one(frames: dict[str, pd.DataFrame]) -> None:
    """Every leak row has label 1 and no rain flag."""
    df = frames[SCENARIO_LEAK]
    assert (df["label"] == 1).all()
    assert (~df["rain_flag"]).all()
    assert (df["scenario"] == SCENARIO_LEAK).all()


def test_rain_not_leak_is_labelled_zero_with_rain_flag(frames: dict[str, pd.DataFrame]) -> None:
    """Every rain_not_leak row has label 0 but rain_flag True."""
    df = frames[SCENARIO_RAIN_NOT_LEAK]
    assert (df["label"] == 0).all()
    assert (df["rain_flag"]).all()
    assert (df["scenario"] == SCENARIO_RAIN_NOT_LEAK).all()


# --- Schema: every clean record matches backend/schemas.py -------------------


@pytest.mark.parametrize("scenario", [SCENARIO_NO_LEAK, SCENARIO_LEAK, SCENARIO_RAIN_NOT_LEAK])
def test_clean_records_validate_against_the_real_esp32_schema(
    frames: dict[str, pd.DataFrame], scenario: str
) -> None:
    """Every non-dropout row passes the same Pydantic model /ingest uses.

    A dropout-affected row is deliberately excluded: it has a field set to
    None to simulate a corrupted transmission, which the real backend would
    reject with HTTP 422 too (see synthetic/dropout.py's docstring). That is
    checked separately in test_dropout_rows_fail_the_real_schema.
    """
    df = frames[scenario]
    clean = df[~df["dropout"]]
    assert len(clean) > 0, "expected at least one clean row to check"

    for row in clean.itertuples():
        payload = row_to_payload(row)
        # Must raise nothing: this is exactly what POST /ingest validates.
        validated = ReadingIn(**payload)
        assert validated.node_id == row.node_id


def test_payload_shape_matches_backend_schema_fields(frames: dict[str, pd.DataFrame]) -> None:
    """The payload built from a row has exactly the real ESP32 payload's fields."""
    df = frames[SCENARIO_NO_LEAK]
    row = df[~df["dropout"]].iloc[0]
    payload = row_to_payload(row)

    assert set(payload.keys()) == set(ReadingIn.model_fields.keys())
    assert set(PAYLOAD_FIELDS) == set(ReadingIn.model_fields.keys())


def test_generated_columns_are_payload_fields_plus_labels(frames: dict[str, pd.DataFrame]) -> None:
    """Every generated column is either a payload field or a documented label."""
    df = frames[SCENARIO_NO_LEAK]
    assert set(df.columns) == set(PAYLOAD_FIELDS) | set(LABEL_FIELDS)


@pytest.mark.parametrize("scenario", [SCENARIO_NO_LEAK, SCENARIO_LEAK, SCENARIO_RAIN_NOT_LEAK])
def test_accel_bursts_have_the_configured_length(
    frames: dict[str, pd.DataFrame], config: Config, scenario: str
) -> None:
    """Every non-dropout accel burst has sensor.accel_burst_samples samples of 3 axes."""
    expected_samples = int(config.get("sensor.accel_burst_samples"))
    df = frames[scenario]
    clean = df[~df["dropout"]]

    for burst in clean["accel"]:
        assert len(burst) == expected_samples
        assert all(len(sample) == 3 for sample in burst)


# --- Dropout ------------------------------------------------------------------


@pytest.mark.parametrize("scenario", [SCENARIO_NO_LEAK, SCENARIO_LEAK, SCENARIO_RAIN_NOT_LEAK])
def test_dropout_rows_have_a_missing_field(
    frames: dict[str, pd.DataFrame], config: Config, scenario: str
) -> None:
    """Every row flagged dropout=True actually has a null candidate field."""
    candidate_fields = list(config.get("synthetic.dropout_fields"))
    df = frames[scenario]
    dropped = df[df["dropout"]]
    assert len(dropped) > 0, "expected at least one dropout row to check"

    for row in dropped.itertuples():
        values = {field: getattr(row, field) for field in candidate_fields}
        missing = [
            field
            for field, value in values.items()
            if value is None or (not isinstance(value, list) and pd.isna(value))
        ]
        assert missing, f"row flagged dropout but no candidate field is missing: {row}"


@pytest.mark.parametrize("scenario", [SCENARIO_NO_LEAK, SCENARIO_LEAK, SCENARIO_RAIN_NOT_LEAK])
def test_dropout_rows_fail_the_real_schema(
    frames: dict[str, pd.DataFrame], scenario: str
) -> None:
    """A dropout-affected row is correctly rejected by the ESP32 payload schema."""
    df = frames[scenario]
    dropped = df[df["dropout"]]
    for row in dropped.itertuples():
        payload = row_to_payload(row)
        with pytest.raises(Exception):
            ReadingIn(**payload)


def test_dropout_rate_is_close_to_configured(frames: dict[str, pd.DataFrame], config: Config) -> None:
    """The observed dropout fraction is in the right ballpark for each scenario."""
    configured = float(config.get("synthetic.dropout_probability"))
    for df in frames.values():
        observed = df["dropout"].mean()
        # Loose tolerance: this is one seeded draw, not a statistical
        # estimate, so the bound just needs to catch a badly wired
        # probability (e.g. off by 10x), not confirm an exact match.
        assert abs(observed - configured) < 0.03, f"dropout rate {observed} far from configured {configured}"


def test_identity_fields_are_never_dropped(frames: dict[str, pd.DataFrame]) -> None:
    """node_id and timestamp survive even on a dropout-affected row."""
    for df in frames.values():
        assert df["node_id"].notna().all()
        assert df["timestamp"].notna().all()


# --- Vibration signature: leak vs. everything else ----------------------------


def _scenario_spectrum(df: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Compute the aggregated (Welch-style) power spectrum for a scenario."""
    sample_rate_hz = float(config.get("sensor.accel_sample_rate_hz"))
    bursts = [burst for burst in df["accel"] if burst is not None]
    return aggregate_power_spectrum(bursts, sample_rate_hz)


def test_leak_dominant_frequency_is_inside_the_configured_band(
    frames: dict[str, pd.DataFrame], config: Config
) -> None:
    """The leak scenario's dominant accelerometer frequency falls in leak_band_hz."""
    band_low, band_high = (float(v) for v in config.get("synthetic.leak_band_hz"))
    freqs, psd = _scenario_spectrum(frames[SCENARIO_LEAK], config)

    peak_freq = dominant_frequency(freqs, psd)
    assert band_low <= peak_freq <= band_high, f"leak peak {peak_freq} Hz outside band [{band_low}, {band_high}]"

    # And it must be a real, persistent tone, not a noise fluke that happens
    # to land in-band: the leak sinusoid should dominate the spectrum.
    ratio = peak_to_mean_ratio(psd)
    assert ratio > 10.0, f"leak peak-to-mean ratio {ratio} too low to be a persistent tone"

    fraction = band_power_fraction(freqs, psd, (band_low, band_high))
    assert fraction > 0.4, f"only {fraction:.2%} of leak vibration power is in-band"


def test_no_leak_has_no_sustained_frequency_component(
    frames: dict[str, pd.DataFrame], config: Config
) -> None:
    """The no_leak scenario's spectrum is flat: no persistent tone anywhere."""
    freqs, psd = _scenario_spectrum(frames[SCENARIO_NO_LEAK], config)

    ratio = peak_to_mean_ratio(psd)
    # A pure-noise spectrum's peak-to-mean ratio sits close to 1; the leak
    # scenario's sits above 10 (see the test above). 5.0 sits comfortably
    # between the two with margin on both sides.
    assert ratio < 5.0, f"no_leak peak-to-mean ratio {ratio} suggests a sustained tone is present"

    band_low, band_high = (float(v) for v in config.get("synthetic.leak_band_hz"))
    fraction = band_power_fraction(freqs, psd, (band_low, band_high))
    assert fraction < 0.3, f"unexpectedly {fraction:.2%} of no_leak vibration power sits in the leak band"


def test_rain_scenario_also_has_no_sustained_vibration(
    frames: dict[str, pd.DataFrame], config: Config
) -> None:
    """Rain causes no persistent vibration either - only moisture/weather signals do."""
    freqs, psd = _scenario_spectrum(frames[SCENARIO_RAIN_NOT_LEAK], config)
    ratio = peak_to_mean_ratio(psd)
    assert ratio < 5.0, f"rain_not_leak peak-to-mean ratio {ratio} suggests a sustained tone is present"


# --- Moisture behaviour: leak drift vs. rain spike -----------------------------


def test_rain_scenario_has_a_sharp_moisture_spike(frames: dict[str, pd.DataFrame], config: Config) -> None:
    """At least one rain event drops soil_raw more than 2x the normal daily swing."""
    baseline_raw = float(config.get("synthetic.soil.no_leak_baseline_raw"))
    daily_amplitude_raw = float(config.get("synthetic.soil.no_leak_daily_amplitude_raw"))

    soil = frames[SCENARIO_RAIN_NOT_LEAK]["soil_raw"].dropna().astype(float)
    spike_depth = baseline_raw - soil.min()

    assert spike_depth > 2.0 * daily_amplitude_raw, (
        f"deepest moisture spike ({spike_depth:.0f} raw counts) does not exceed "
        f"2x the normal daily swing ({2 * daily_amplitude_raw:.0f})"
    )


def test_leak_scenario_soil_moisture_trends_wetter(frames: dict[str, pd.DataFrame]) -> None:
    """The leak scenario's soil_raw falls (gets wetter) from start to end."""
    soil = frames[SCENARIO_LEAK]["soil_raw"].dropna().astype(float)
    first_hour_mean = soil.iloc[:60].mean()
    last_hour_mean = soil.iloc[-60:].mean()
    assert last_hour_mean < first_hour_mean, "leak scenario did not get wetter over time"


def test_leak_scenario_starts_near_the_no_leak_baseline(frames: dict[str, pd.DataFrame], config: Config) -> None:
    """The leak scenario starts at the configured baseline, before it drifts.

    Unlike no_leak/rain, the leak soil model (see
    synthetic.scenarios.soil_leak) has no daily cycle term, only a starting
    baseline plus noise - so, unlike those two, its first reading should sit
    tightly around the configured baseline regardless of what hour the run
    starts at.
    """
    baseline_raw = float(config.get("synthetic.soil.no_leak_baseline_raw"))
    first_leak = frames[SCENARIO_LEAK]["soil_raw"].dropna().astype(float).iloc[0]

    tolerance = float(config.get("synthetic.soil.noise_sigma_raw")) * 5
    assert abs(first_leak - baseline_raw) < tolerance


# --- Reproducibility -----------------------------------------------------------


@pytest.mark.parametrize("scenario", [SCENARIO_NO_LEAK, SCENARIO_LEAK, SCENARIO_RAIN_NOT_LEAK])
def test_seeded_runs_are_identical(config: Config, scenario: str) -> None:
    """The same seed always produces byte-for-byte identical output."""
    first = generate_scenario(scenario, config, seed=123)
    second = generate_scenario(scenario, config, seed=123)

    non_list_columns = [c for c in first.columns if c not in ("accel", "mag")]
    assert first[non_list_columns].equals(second[non_list_columns])
    assert list(first["accel"]) == list(second["accel"])
    assert list(first["mag"]) == list(second["mag"])


def test_different_seeds_produce_different_data(config: Config) -> None:
    """Different seeds do not coincidentally produce the same run."""
    first = generate_scenario(SCENARIO_LEAK, config, seed=1)
    second = generate_scenario(SCENARIO_LEAK, config, seed=2)
    assert not first["soil_raw"].astype(float).equals(second["soil_raw"].astype(float))


def test_generate_all_uses_a_derived_seed_per_scenario(config: Config, seed: int) -> None:
    """generate_all's per-scenario draws differ from each other (not one repeated stream)."""
    frames_run = generate_all(config, seed=seed)
    no_leak_soil = frames_run[SCENARIO_NO_LEAK]["soil_raw"].astype(float).to_numpy()
    leak_soil = frames_run[SCENARIO_LEAK]["soil_raw"].astype(float).to_numpy()
    assert not np.array_equal(no_leak_soil, leak_soil)


# --- Combining scenarios --------------------------------------------------------


def test_combine_scenarios_concatenates_every_row(frames: dict[str, pd.DataFrame]) -> None:
    """The combined frame has exactly as many rows as all scenarios summed."""
    combined = combine_scenarios(frames)
    assert len(combined) == sum(len(df) for df in frames.values())
    assert set(combined["scenario"].unique()) == set(frames.keys())


# --- Parquet round trip ----------------------------------------------------------


def test_save_and_load_single_scenario_round_trips(tmp_path, config: Config, seed: int) -> None:
    """A saved-then-loaded scenario keeps its schema, values and dropout flags."""
    df = generate_scenario(SCENARIO_LEAK, config, seed=seed)
    path = tmp_path / "leak.parquet"
    save_scenario(df, path)
    loaded = load_scenario(path)

    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)
    assert loaded["dropout"].sum() == df["dropout"].sum()

    # Clean rows must still validate after a save/load cycle.
    clean = loaded[~loaded["dropout"]]
    for row in clean.head(20).itertuples():
        ReadingIn(**row_to_payload(row))

    # A dropout row's burst/scalar fields must still come back as None, not NaN.
    dropped = loaded[loaded["dropout"]].iloc[0]
    payload = row_to_payload(dropped)
    assert any(value is None for value in payload.values())


def test_save_all_writes_per_scenario_and_combined_files(tmp_path, frames: dict[str, pd.DataFrame]) -> None:
    """save_all() writes one file per scenario plus all_scenarios.parquet."""
    written = save_all(frames, tmp_path)

    for name in frames:
        assert written[name].exists()
    combined_path = tmp_path / "all_scenarios.parquet"
    assert written["all"] == combined_path
    assert combined_path.exists()

    combined = load_scenario(combined_path)
    assert len(combined) == sum(len(df) for df in frames.values())


# --- Configuration is actually driving the output (nothing hardcoded) ----------


def test_soil_baseline_follows_configuration(config: Config) -> None:
    """Changing the configured no-leak baseline changes the generated data."""
    from backend.config import Config as ConfigClass

    baseline_key = "synthetic.soil.no_leak_baseline_raw"
    original_baseline = float(config.get(baseline_key))

    patched_data = config.as_dict()
    patched_data["synthetic"] = dict(patched_data["synthetic"])
    patched_data["synthetic"]["soil"] = dict(patched_data["synthetic"]["soil"])
    patched_data["synthetic"]["soil"]["no_leak_baseline_raw"] = original_baseline + 500.0
    patched_config = ConfigClass(patched_data, source_path=config.source_path)

    df_original = generate_scenario(SCENARIO_NO_LEAK, config, seed=1)
    df_patched = generate_scenario(SCENARIO_NO_LEAK, patched_config, seed=1)

    diff = df_patched["soil_raw"].astype(float).mean() - df_original["soil_raw"].astype(float).mean()
    assert diff == pytest.approx(500.0, abs=5.0)


def test_leak_band_change_moves_the_dominant_frequency(config: Config) -> None:
    """Changing the configured leak band changes where the vibration tone sits."""
    from backend.config import Config as ConfigClass

    patched_data = config.as_dict()
    patched_data["synthetic"] = dict(patched_data["synthetic"])
    patched_data["synthetic"]["leak_band_hz"] = [150.0, 170.0]
    patched_config = ConfigClass(patched_data, source_path=config.source_path)

    df = generate_scenario(SCENARIO_LEAK, patched_config, seed=1)
    sample_rate_hz = float(config.get("sensor.accel_sample_rate_hz"))
    bursts = [b for b in df["accel"] if b is not None]
    freqs, psd = aggregate_power_spectrum(bursts, sample_rate_hz)
    peak_freq = dominant_frequency(freqs, psd)

    assert 150.0 <= peak_freq <= 170.0
