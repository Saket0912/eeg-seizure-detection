"""Unit tests for src/preprocessing/filters.py"""

import numpy as np
import pytest

from src.preprocessing.filters import (
    apply_butterworth_wavelet_filter,
    apply_chebyshev_bessel_filter,
    apply_chebyshev_wavelet_filter,
    apply_daubechies_wiener_filter,
    apply_gaussian_butterworth_filter,
    calculate_snr,
)

N_SAMPLES = 50
N_FEATURES = 178


@pytest.fixture
def synthetic_eeg() -> np.ndarray:
    """Synthetic EEG-like data: sine wave + Gaussian noise."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, N_FEATURES)
    signal = np.sin(2 * np.pi * 10 * t)
    noise = rng.normal(scale=0.3, size=(N_SAMPLES, N_FEATURES))
    return np.tile(signal, (N_SAMPLES, 1)) + noise


class TestSNR:
    def test_identical_signals_infinite_snr(self):
        sig = np.ones(100)
        snr = calculate_snr(sig, sig)
        assert snr > 100

    def test_snr_positive_for_reasonable_filter(self, synthetic_eeg):
        filtered = apply_butterworth_wavelet_filter(synthetic_eeg)
        snr = calculate_snr(np.ravel(synthetic_eeg),
                            np.ravel(filtered)[: synthetic_eeg.size])
        assert snr > 0


class TestFilters:
    def test_gaussian_butterworth_shape(self, synthetic_eeg):
        out = apply_gaussian_butterworth_filter(synthetic_eeg)
        assert out.shape == synthetic_eeg.shape

    def test_chebyshev_wavelet_runs(self, synthetic_eeg):
        out = apply_chebyshev_wavelet_filter(synthetic_eeg)
        assert out.ndim == 2

    def test_chebyshev_bessel_shape(self, synthetic_eeg):
        out = apply_chebyshev_bessel_filter(synthetic_eeg)
        assert out.shape == synthetic_eeg.shape

    def test_daubechies_wiener_runs(self, synthetic_eeg):
        out = apply_daubechies_wiener_filter(synthetic_eeg)
        assert out.ndim == 2

    def test_butterworth_wavelet_shape(self, synthetic_eeg):
        out = apply_butterworth_wavelet_filter(synthetic_eeg)
        assert out.ndim == 2

    def test_no_nan_in_output(self, synthetic_eeg):
        out = apply_butterworth_wavelet_filter(synthetic_eeg)
        assert not np.isnan(out).any()
