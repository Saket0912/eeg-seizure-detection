"""Unit tests for visualization utilities (smoke tests — no display)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.utils.visualization import (
    plot_filtered_eeg,
    plot_model_comparison,
    plot_raw_eeg,
    plot_snr_comparison,
)


@pytest.fixture(autouse=True)
def close_plots():
    """Ensure matplotlib figures are closed after every test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    cols = [f"X{i}" for i in range(1, 179)]
    data = np.random.randn(10, 178)
    return pd.DataFrame(data, columns=cols)


def test_plot_raw_eeg_runs(sample_df):
    plot_raw_eeg(sample_df, n_rows=3)


def test_plot_filtered_eeg_runs(sample_df):
    plot_filtered_eeg(sample_df, n_rows=3)


def test_plot_snr_comparison_runs():
    plot_snr_comparison(
        filter_names=["Filter A", "Filter B", "Filter C"],
        snr_values=[10.5, 14.2, 12.8],
    )


def test_plot_model_comparison_runs():
    plot_model_comparison(
        model_names=["AlexNet", "GoogLeNet"],
        metrics={
            "Accuracy":  [0.92, 0.97],
            "Precision": [0.92, 0.97],
            "Recall":    [0.92, 0.97],
            "F1-Score":  [0.92, 0.97],
        },
    )
