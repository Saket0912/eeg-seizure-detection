"""
EEG Signal Preprocessing Filters
==================================
Implements five filtering strategies compared by SNR:

1. Gaussian + Butterworth
2. Chebyshev + Wavelet Denoising
3. Chebyshev + Bessel
4. Daubechies + Wiener
5. Butterworth + Wavelet Denoising  ← Best SNR (selected pipeline)
"""

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, cheby1, bessel, wiener
from typing import Tuple, Dict, Any


def calculate_snr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.

    Parameters
    ----------
    original : np.ndarray
        Original (unfiltered) signal, flattened 1-D array.
    filtered : np.ndarray
        Filtered signal, same shape as *original*.

    Returns
    -------
    float
        SNR value in dB.
    """
    noise = original - filtered
    snr = 10 * np.log10(np.sum(original ** 2) / (np.sum(noise ** 2) + 1e-10))
    return float(snr)


def apply_gaussian_filter(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply a Gaussian smoothing filter."""
    return gaussian_filter(data, sigma=sigma)


def apply_butter_lowpass(
    data: np.ndarray,
    cutoff_freq: float,
    sampling_freq: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def apply_gaussian_butterworth_filter(
    data: np.ndarray,
    sigma: float = 2.0,
    cutoff_freq: float = 60.0,
    sampling_freq: float = 256.0,
    order: int = 2,
) -> np.ndarray:
    """Combined Gaussian + Butterworth filter (average of both outputs)."""
    gaussian_filtered = apply_gaussian_filter(data, sigma)
    butter_filtered = apply_butter_lowpass(data, cutoff_freq, sampling_freq, order)
    return (gaussian_filtered + butter_filtered) / 2.0


def apply_chebyshev_lowpass(
    data: np.ndarray,
    cutoff_freq: float,
    sampling_freq: float,
    ripple_db: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Apply a Chebyshev Type-I low-pass filter."""
    nyquist = 0.5 * sampling_freq
    wn = cutoff_freq / nyquist
    b, a = cheby1(order, ripple_db, wn, btype="low", analog=False)
    return filtfilt(b, a, data)


def apply_wavelet_denoising(
    data: np.ndarray,
    wavelet: str = "db4",
    threshold: float = 1.0,
) -> np.ndarray:
    """Denoise signal using Discrete Wavelet Transform + soft thresholding."""
    coeffs = pywt.wavedec(data, wavelet)
    coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


def apply_chebyshev_wavelet_filter(
    data: np.ndarray,
    cutoff_freq: float = 60.0,
    sampling_freq: float = 256.0,
    ripple_db: float = 0.5,
    order: int = 4,
    wavelet: str = "db4",
    threshold: float = 1.0,
) -> np.ndarray:
    """Combined Chebyshev low-pass + wavelet denoising pipeline."""
    cheby_filtered = apply_chebyshev_lowpass(
        data, cutoff_freq, sampling_freq, ripple_db, order
    )
    return apply_wavelet_denoising(cheby_filtered, wavelet, threshold)


def apply_bessel_lowpass(
    data: np.ndarray,
    cutoff_freq: float,
    sampling_freq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Bessel low-pass filter (maximally flat group delay)."""
    nyquist = 0.5 * sampling_freq
    wn = cutoff_freq / nyquist
    b, a = bessel(order, wn, btype="low", analog=False)
    return filtfilt(b, a, data)


def apply_chebyshev_bessel_filter(
    data: np.ndarray,
    cutoff_freq: float = 60.0,
    sampling_freq: float = 256.0,
    ripple_db: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Combined Chebyshev + Bessel filter (average of both outputs)."""
    cheby = apply_chebyshev_lowpass(data, cutoff_freq, sampling_freq, ripple_db, order)
    bess = apply_bessel_lowpass(data, cutoff_freq, sampling_freq, order)
    return (cheby + bess) / 2.0


def apply_daubechies_wiener_filter(
    data: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold: float = 1.0,
) -> np.ndarray:
    """Wiener filter followed by Daubechies wavelet denoising."""
    wiener_filtered = wiener(data)
    coeffs = pywt.wavedec(wiener_filtered, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


def apply_butterworth_wavelet_filter(
    data: np.ndarray,
    cutoff_freq: float = 40.0,
    sampling_freq: float = 256.0,
    wavelet: str = "db4",
    level: int = 4,
    order: int = 5,
) -> np.ndarray:
    """
    **Selected pipeline** — Butterworth low-pass + wavelet denoising.

    Achieves the highest SNR across both EEG datasets and is used
    as the preprocessing step for all downstream models.
    """
    butter_filtered = apply_butter_lowpass(data, cutoff_freq, sampling_freq, order)
    coeffs = pywt.wavedec(butter_filtered, wavelet, level=level)
    threshold = 1.0
    coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


def compare_all_filters(
    data: np.ndarray,
    sampling_freq: float = 256.0,
) -> Dict[str, Dict[str, Any]]:
    """Apply all five filter pipelines and return SNR for each."""
    original = np.ravel(data)

    filters = {
        "Gaussian + Butterworth": apply_gaussian_butterworth_filter(
            data, sampling_freq=sampling_freq
        ),
        "Chebyshev + Wavelet": apply_chebyshev_wavelet_filter(
            data, sampling_freq=sampling_freq
        ),
        "Chebyshev + Bessel": apply_chebyshev_bessel_filter(
            data, sampling_freq=sampling_freq
        ),
        "Daubechies + Wiener": apply_daubechies_wiener_filter(data),
        "Butterworth + Wavelet (Selected)": apply_butterworth_wavelet_filter(
            data, sampling_freq=sampling_freq
        ),
    }

    results = {}
    for name, filtered in filters.items():
        snr = calculate_snr(original, np.ravel(filtered)[: len(original)])
        results[name] = {"filtered_data": filtered, "snr_db": snr}
        print(f"  {name:35s} → SNR: {snr:6.2f} dB")

    return results
