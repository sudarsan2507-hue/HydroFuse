"""Vibration features extracted from accelerometer bursts.

Every function is pure: plain arrays/lists and numbers in, numbers out - no
config, no I/O, no global state - so each is independently unit-testable and
behaves identically whether the burst came from a real ESP32 or the
synthetic generator.

Power spectra are combined across the three axes by summing power (squared
FFT magnitude), never by rectifying a time-domain signal (e.g. taking a
vector magnitude first) - rectifying folds a negative half-cycle onto the
positive one and can double the apparent frequency of a clean sinusoid.
"""

from __future__ import annotations

import numpy as np


def _burst_array(burst: list[list[float]]) -> np.ndarray:
    """Convert a ``[[ax, ay, az], ...]`` burst into an ``(n, 3)`` float array."""
    return np.asarray(burst, dtype=np.float64)


def filter_transients(burst: list[list[float]], sigma_threshold: float) -> np.ndarray:
    """Remove footstep-like transient spikes from a burst before analysis.

    A sample is treated as a transient if any axis's deviation from that
    axis's own mean exceeds ``sigma_threshold`` standard deviations. Flagged
    samples are replaced with their axis's mean (not dropped, so the burst
    keeps a constant length and even sample spacing for the FFT).

    Args:
        burst: The accelerometer burst.
        sigma_threshold: Number of standard deviations beyond which a sample
            is treated as a transient spike.

    Returns:
        An ``(n, 3)`` array with transient samples replaced by the per-axis
        mean.
    """
    samples = _burst_array(burst)
    mean = samples.mean(axis=0, keepdims=True)
    std = samples.std(axis=0, keepdims=True)
    std_safe = np.where(std == 0.0, 1.0, std)  # avoid divide-by-zero on a flat axis

    deviation = np.abs(samples - mean) / std_safe
    is_transient = np.any(deviation > sigma_threshold, axis=1)

    filtered = samples.copy()
    filtered[is_transient] = mean
    return filtered


def combined_power_spectrum(samples: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Sum the per-axis power spectra of an already-array-shaped burst.

    Args:
        samples: An ``(n, 3)`` array (e.g. from :func:`filter_transients`).
        sample_rate_hz: Sampling rate the burst was captured at.

    Returns:
        A tuple ``(freqs_hz, power)``; bin 0 is DC.
    """
    centred = samples - samples.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centred, axis=0)
    power = np.sum(np.abs(spectrum) ** 2, axis=1)
    freqs = np.fft.rfftfreq(samples.shape[0], d=1.0 / sample_rate_hz)
    return freqs, power


def aggregate_bursts_power_spectrum(
    bursts: list[list[list[float]]],
    sample_rate_hz: float,
    transient_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter transients out of and then average the power spectrum of several bursts.

    Args:
        bursts: A list of equal-length accelerometer bursts (e.g. every
            reading inside one feature window).
        sample_rate_hz: Sampling rate the bursts were captured at.
        transient_sigma: Threshold passed to :func:`filter_transients`.

    Returns:
        A tuple ``(freqs_hz, mean_power)``.

    Raises:
        ValueError: If ``bursts`` is empty.
    """
    if not bursts:
        raise ValueError("aggregate_bursts_power_spectrum() needs at least one burst")

    total_power: np.ndarray | None = None
    freqs: np.ndarray | None = None
    for burst in bursts:
        filtered = filter_transients(burst, transient_sigma)
        burst_freqs, power = combined_power_spectrum(filtered, sample_rate_hz)
        if total_power is None:
            total_power = power
            freqs = burst_freqs
        else:
            total_power = total_power + power
    assert freqs is not None and total_power is not None
    return freqs, total_power / len(bursts)


def dominant_frequency(freqs_hz: np.ndarray, power: np.ndarray) -> float:
    """Return the non-DC frequency with the most power.

    Args:
        freqs_hz: Frequency bins.
        power: Power at each bin.

    Returns:
        The frequency, in Hz, of the largest non-DC bin (0.0 if there is
        only a DC bin).
    """
    if len(power) <= 1:
        return 0.0
    peak_index = int(np.argmax(power[1:])) + 1
    return float(freqs_hz[peak_index])


def band_power(freqs_hz: np.ndarray, power: np.ndarray, band_hz: tuple[float, float]) -> float:
    """Return the total (non-normalised) power inside a frequency band.

    Args:
        freqs_hz: Frequency bins.
        power: Power at each bin.
        band_hz: ``(low, high)`` band edges in Hz, inclusive.

    Returns:
        The sum of power in bins whose frequency falls inside the band.
    """
    low, high = band_hz
    in_band = (freqs_hz >= low) & (freqs_hz <= high)
    return float(np.sum(power[in_band]))


def spectral_entropy(power: np.ndarray) -> float:
    """Return the Shannon entropy of the (non-DC) normalised power spectrum.

    A single strong, persistent tone (a leak) concentrates power into a few
    bins and has low entropy; broadband noise spreads power evenly and has
    high entropy. The value is normalised to ``[0, 1]`` by dividing by
    ``log2(n_bins)``, so it does not depend on burst length.

    Args:
        power: Power at each frequency bin (bin 0 assumed to be DC and
            excluded).

    Returns:
        Normalised spectral entropy in ``[0, 1]``, or 0.0 if there is no
        non-DC power at all.
    """
    non_dc = power[1:]
    total = float(np.sum(non_dc))
    if total <= 0.0 or len(non_dc) < 2:
        return 0.0

    probabilities = non_dc / total
    # 0 * log(0) is defined as 0 for entropy purposes.
    nonzero = probabilities[probabilities > 0.0]
    raw_entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    max_entropy = np.log2(len(non_dc))
    return raw_entropy / max_entropy if max_entropy > 0.0 else 0.0


def vibration_rms(samples: np.ndarray) -> float:
    """Return the RMS vibration amplitude of a burst, with gravity/DC removed.

    Args:
        samples: An ``(n, 3)`` array (e.g. from :func:`filter_transients`).

    Returns:
        RMS of the mean-removed signal, combined across all three axes.
    """
    centred = samples - samples.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))


def vibration_features(
    bursts: list[list[list[float]]],
    sample_rate_hz: float,
    leak_band_hz: tuple[float, float],
    transient_sigma: float,
) -> dict[str, float]:
    """Compute the full set of vibration features for a group of bursts.

    Args:
        bursts: Accelerometer bursts to summarise together (typically every
            reading in one feature window).
        sample_rate_hz: Sampling rate the bursts were captured at.
        leak_band_hz: ``(low, high)`` band edges to compute band power in.
        transient_sigma: Threshold used to filter footstep-like spikes before
            analysis.

    Returns:
        A dict with ``dominant_freq_hz``, ``leak_band_power``,
        ``spectral_entropy`` and ``vibration_rms``.
    """
    freqs, power = aggregate_bursts_power_spectrum(bursts, sample_rate_hz, transient_sigma)
    rms_values = [vibration_rms(filter_transients(b, transient_sigma)) for b in bursts]

    return {
        "dominant_freq_hz": dominant_frequency(freqs, power),
        "leak_band_power": band_power(freqs, power, leak_band_hz),
        "spectral_entropy": spectral_entropy(power),
        "vibration_rms": float(np.mean(rms_values)),
    }
