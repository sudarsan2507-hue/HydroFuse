r"""Generate the three labelled synthetic scenarios and sanity-check them.

Produces, from ``config.yaml`` alone (nothing hardcoded):

* ``data/synthetic/no_leak.parquet``
* ``data/synthetic/leak.parquet``
* ``data/synthetic/rain_not_leak.parquet``
* ``data/synthetic/all_scenarios.parquet`` (all three, with a ``scenario`` column)
* ``outputs/synthetic_soil_plot.png`` - soil_raw over time, all three scenarios
* a console summary: mean soil moisture, accel vibration RMS, and the
  dominant accelerometer frequency for each scenario, so the leak's
  vibration signature can be confirmed by eye before Phase 5 builds features
  on top of it.

Run it with::

    .venv\Scripts\python scripts\generate_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Config, load_config  # noqa: E402
from synthetic.analysis import (  # noqa: E402
    aggregate_power_spectrum,
    band_power_fraction,
    dominant_frequency,
    peak_to_mean_ratio,
    vibration_rms,
)
from synthetic.generator import generate_all  # noqa: E402
from synthetic.io import save_all  # noqa: E402

# Fixed categorical colours, assigned by identity (never re-cycled), from the
# project's validated three-series palette.
_SCENARIO_COLORS = {
    "no_leak": "#2a78d6",  # blue
    "leak": "#eb6834",  # orange
    "rain_not_leak": "#1baf7a",  # aqua
}
_GRID_COLOR = "#d9d8d3"
_TEXT_COLOR = "#0b0b0b"


def scenario_summary(df: pd.DataFrame, config: Config) -> dict[str, float]:
    """Compute the sanity-check metrics for one scenario's DataFrame.

    Args:
        df: A scenario DataFrame as produced by
            :func:`synthetic.generator.generate_scenario`.
        config: Project configuration, for the sample rate and leak band.

    Returns:
        A dictionary with ``mean_soil_raw``, ``accel_rms_g``,
        ``dominant_freq_hz``, ``leak_band_power_fraction`` and
        ``peak_to_mean_ratio``.
    """
    sample_rate_hz = float(config.get("sensor.accel_sample_rate_hz"))
    band_hz = tuple(float(v) for v in config.get("synthetic.leak_band_hz"))

    # Dropout-affected rows have accel=None; they carry no vibration signal
    # to analyse, so they are skipped here (they are still counted and
    # reported separately - see main()).
    bursts = [burst for burst in df["accel"] if burst is not None]

    freqs, psd = aggregate_power_spectrum(bursts, sample_rate_hz)

    return {
        "mean_soil_raw": float(df["soil_raw"].dropna().astype(float).mean()),
        "accel_rms_g": float(pd.Series([vibration_rms(b) for b in bursts]).mean()),
        "dominant_freq_hz": dominant_frequency(freqs, psd),
        "leak_band_power_fraction": band_power_fraction(freqs, psd, band_hz),
        "peak_to_mean_ratio": peak_to_mean_ratio(psd),
    }


def plot_soil_moisture(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot soil_raw over time for every scenario on one chart.

    Args:
        frames: Mapping from scenario name to its DataFrame.
        output_path: Where to save the PNG. Parent directory is created if
            it does not exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, df in frames.items():
        color = _SCENARIO_COLORS.get(name, "#4a3aa7")
        # A dropped soil_raw shows as a gap rather than a fake zero: pandas
        # already carries pandas.NA/None through this column, and
        # matplotlib skips NaN points in a line, leaving a visible break.
        soil = df["soil_raw"].astype("Float64")
        ax.plot(
            df["timestamp"],
            soil.to_numpy(dtype=float, na_value=float("nan")),
            label=name,
            color=color,
            linewidth=1.5,
        )

    ax.set_title("Synthetic soil moisture (raw ADC, lower = wetter)", color=_TEXT_COLOR, fontsize=13)
    ax.set_xlabel("Time", color=_TEXT_COLOR)
    ax.set_ylabel("soil_raw (ADC counts)", color=_TEXT_COLOR)
    ax.tick_params(colors=_TEXT_COLOR)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> int:
    """Generate, save, plot and summarise all configured scenarios.

    Returns:
        Process exit code (always 0; this script is diagnostic, not a check).
    """
    config = load_config()
    seed = int(config.get("project.random_seed"))

    print(f"Generating synthetic scenarios (seed={seed}) ...")
    frames = generate_all(config, seed=seed)

    output_dir = config.resolve_path(config.get("synthetic.output_dir"))
    written = save_all(frames, output_dir)
    print(f"Wrote {len(frames)} scenario file(s) + combined file to {output_dir}:")
    for name, path in written.items():
        print(f"  {name:15s} -> {path}")

    plot_path = config.resolve_path("outputs/synthetic_soil_plot.png")
    plot_soil_moisture(frames, plot_path)
    print(f"\nSaved sanity plot to {plot_path}")

    print("\nSanity check (dropout-affected bursts excluded from the accel stats):")
    header = f"{'scenario':15s} {'rows':>6s} {'dropout':>8s} {'mean_soil_raw':>14s} {'accel_rms_g':>12s} {'dominant_hz':>12s} {'in_leak_band':>13s} {'peak/mean':>10s}"
    print(header)
    print("-" * len(header))

    band_hz = tuple(float(v) for v in config.get("synthetic.leak_band_hz"))
    for name, df in frames.items():
        stats = scenario_summary(df, config)
        in_band = band_hz[0] <= stats["dominant_freq_hz"] <= band_hz[1]
        print(
            f"{name:15s} {len(df):6d} {int(df['dropout'].sum()):8d} "
            f"{stats['mean_soil_raw']:14.1f} {stats['accel_rms_g']:12.5f} "
            f"{stats['dominant_freq_hz']:12.2f} {str(in_band):>13s} {stats['peak_to_mean_ratio']:10.2f}"
        )

    print(f"\nConfigured leak band: {band_hz[0]:.1f}-{band_hz[1]:.1f} Hz")
    print(
        "Expect: 'leak' dominant frequency inside the band with a high peak/mean "
        "ratio (a persistent tone); 'no_leak' and 'rain_not_leak' near peak/mean ~1 "
        "(no persistent tone - noise only)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
