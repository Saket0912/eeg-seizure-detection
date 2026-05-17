"""
EEG Signal Preprocessing Module
================================
Signal filtering and noise reduction utilities for EEG data.
"""

from .filters import (
    apply_butterworth_wavelet_filter,
    apply_gaussian_butterworth_filter,
    apply_chebyshev_wavelet_filter,
    apply_chebyshev_bessel_filter,
    apply_daubechies_wiener_filter,
    calculate_snr,
    compare_all_filters,
)

__all__ = [
    "apply_butterworth_wavelet_filter",
    "apply_gaussian_butterworth_filter",
    "apply_chebyshev_wavelet_filter",
    "apply_chebyshev_bessel_filter",
    "apply_daubechies_wiener_filter",
    "calculate_snr",
    "compare_all_filters",
]
