"""Small signal-analysis helpers used to sanity-check the synthetic data.

These are deliberately simple, pure functions: given a burst (or a list of
bursts) they compute a vibration RMS, a power spectrum and a dominant
frequency. Phase 5 will build a fuller feature-extraction pipeline; this
module exists so Phase 3 can prove to itself (and to tests) that the leak
signature is actually visible in the generated data, without waiting for
Phase 5 to exist.

Frequency content is computed from the *power spectrum* (squared FFT
magnitude), not the raw time-domain signal. Combining axes by summing power
avoids a subtle bug: rectifying (e.g. taking a vector magnitude) a sinusoid
that dips negative folds it onto the positive axis and doubles its apparent
frequency. Summing power spectra has no such artifact.
"""

from __future__ import annotations

import numpy as np


def _burst_to_array(burst: list[list[float]]) -> np.ndarray:
    """Convert a ``[[ax, ay, az], ...]`` burst into an ``(n, 3)`` array.

    Args:
        burst: The accelerometer burst.

    Returns:
        A float64 array of shape ``(n_samples, 3)``.
    """
    return np.asarray(burst, dtype=np.float64)


def vibration_rms(burst: list[list[float]]) -> float:
    """Return the RMS vibration amplitude of one burst, with gravity removed.

    Each axis's own mean (which includes the static gravity component, e.g.
    ~1g on Z) is subtracted before combining, so this measures how much the
    signal *moves*, not where it sits.

    Args:
        burst: The accelerometer burst.

    Returns:
        RMS of the mean-removed signal, combined across all three axes, in g.
    """
    samples = _burst_to_array(burst)
    centred = samples - samples.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))


def burst_power_spectrum(burst: list[list[float]], sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the combined (summed across axes) power spectrum of one burst.

    Args:
        burst: The accelerometer burst.
        sample_rate_hz: Sampling rate the burst was captured at.

    Returns:
        A tuple ``(freqs_hz, power)``, both arrays of the same length, where
        ``power[i]`` is the total power at ``freqs_hz[i]`` summed over the
        three axes. Bin 0 is DC (0 Hz).
    """
    samples = _burst_to_array(burst)
    centred = samples - samples.mean(axis=0, keepdims=True)
    n = centred.shape[0]

    spectrum = np.fft.rfft(centred, axis=0)  # shape (n_freqs, 3)
    power = np.sum(np.abs(spectrum) ** 2, axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    return freqs, power


def aggregate_power_spectrum(
    bursts: list[list[list[float]]], sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average the power spectrum of many equal-length bursts.

    Averaging power spectra (a Welch-style periodogram average) is what lets
    a persistent, coherent signal (the leak's sinusoid) stand out above
    background noise that averages toward a flat spectrum over many bursts.

    Args:
        bursts: A list of accelerometer bursts, all the same length.
        sample_rate_hz: Sampling rate the bursts were captured at.

    Returns:
        A tuple ``(freqs_hz, mean_power)``.

    Raises:
        ValueError: If ``bursts`` is empty.
    """
    if not bursts:
        raise ValueError("aggregate_power_spectrum() needs at least one burst")

    freqs, first_power = burst_power_spectrum(bursts[0], sample_rate_hz)
    total = first_power.copy()
    for burst in bursts[1:]:
        _, power = burst_power_spectrum(burst, sample_rate_hz)
        total += power
    return freqs, total / len(bursts)


def dominant_frequency(freqs_hz: np.ndarray, power: np.ndarray) -> float:
    """Return the frequency with the most power, ignoring DC.

    Args:
        freqs_hz: Frequency bins.
        power: Power at each bin, same length as ``freqs_hz``.

    Returns:
        The frequency, in Hz, of the largest non-DC bin.
    """
    if len(power) <= 1:
        return 0.0
    peak_index = int(np.argmax(power[1:])) + 1
    return float(freqs_hz[peak_index])


def band_power_fraction(freqs_hz: np.ndarray, power: np.ndarray, band_hz: tuple[float, float]) -> float:
    """Return the fraction of (non-DC) power that falls inside a frequency band.

    Args:
        freqs_hz: Frequency bins.
        power: Power at each bin, same length as ``freqs_hz``.
        band_hz: ``(low, high)`` band edges in Hz, inclusive.

    Returns:
        A value in ``[0, 1]``. 0 if there is no non-DC power at all.
    """
    non_dc_power = power[1:]
    total = float(np.sum(non_dc_power))
    if total <= 0.0:
        return 0.0
    low, high = band_hz
    in_band = (freqs_hz[1:] >= low) & (freqs_hz[1:] <= high)
    return float(np.sum(non_dc_power[in_band]) / total)


def peak_to_mean_ratio(power: np.ndarray) -> float:
    """Return how many times larger the strongest non-DC bin is than average.

    A flat, noise-like spectrum has a ratio close to 1; a single strong
    persistent tone (a leak) produces a large ratio.

    Args:
        power: Power at each frequency bin (bin 0 assumed to be DC).

    Returns:
        The peak-to-mean ratio, or 0.0 if there is no non-DC power.
    """
    non_dc = power[1:]
    mean_power = float(np.mean(non_dc)) if len(non_dc) else 0.0
    if mean_power <= 0.0:
        return 0.0
    return float(np.max(non_dc)) / mean_power
