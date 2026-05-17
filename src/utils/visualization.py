"""
Visualization Utilities
========================
All plotting functions for the EEG Seizure Detection pipeline.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def plot_raw_eeg(
    X: pd.DataFrame,
    n_rows: int = 5,
    title: str = "Raw EEG Signals (First 5 Samples)",
    save_path: Optional[str] = None,
) -> None:
    """Plot raw EEG waveforms for the first *n_rows* samples."""
    eeg_cols = [f"X{i}" for i in range(1, 179)]
    plt.figure(figsize=(10, 4))
    for i in range(min(n_rows, len(X))):
        plt.plot(X.iloc[i][eeg_cols].astype(float).values,
                 label=f"Sample {i + 1}", alpha=0.8)
    plt.xlabel("Time Point")
    plt.ylabel("EEG Signal Intensity (µV)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_filtered_eeg(
    filtered_df: pd.DataFrame,
    n_rows: int = 5,
    title: str = "Filtered EEG Signals (First 5 Samples)",
    save_path: Optional[str] = None,
) -> None:
    """Plot filtered EEG waveforms for the first *n_rows* samples."""
    eeg_cols = [f"X{i}" for i in range(1, filtered_df.shape[1] + 1)]
    plt.figure(figsize=(10, 4))
    for i in range(min(n_rows, len(filtered_df))):
        plt.plot(filtered_df.iloc[i][eeg_cols].astype(float).values,
                 label=f"Sample {i + 1}", alpha=0.8)
    plt.xlabel("Time Point")
    plt.ylabel("EEG Signal Intensity (µV)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_snr_comparison(
    filter_names: List[str],
    snr_values: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing SNR across different filter pipelines."""
    
    colors = ['#A8DADC', '#F4A261', '#A7C957', '#E9C46A', '#E76F51']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(filter_names, snr_values, color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.95)
    
    # Add value labels
    for bar, snr in zip(bars, snr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{snr:.2f} dB', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Filter Pipeline', fontsize=13, fontweight='bold')
    ax.set_ylabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=13, fontweight='bold')
    ax.set_title('EEG Signal Preprocessing - Filter SNR Comparison',
                  fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, max(snr_values) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    
    # Highlight best filter
    best_idx = np.argmax(snr_values)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_alpha(1.0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_training_curves(
    history,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss / accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",   linewidth=2,
                 linestyle="--")
    axes[0].set_title(f"{model_name} — Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history.history["accuracy"],     label="Train Acc", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc",   linewidth=2,
                 linestyle="--")
    axes[1].set_title(f"{model_name} — Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing all models across four metrics."""
    
    # Professional color palette
    colors = ['#2E86AB', '#F6BD60', '#F77D0B', '#8338EC', '#3A86FF', '#FB5607']
    
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    n_models = len(model_names)
    bar_width = 0.14
    
    fig, ax = plt.subplots(figsize=(14, 7))
    index = np.arange(n_metrics)
    
    for i, (model, color) in enumerate(zip(model_names, colors)):
        values = [metrics[m][i] for m in metric_names]
        bars = ax.bar(index + i * bar_width, values, bar_width,
               label=model, color=color, edgecolor='black', linewidth=1.2,
               alpha=0.95)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', rotation=0)
    
    ax.set_xlabel('Performance Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Scores', fontsize=13, fontweight='bold')
    ax.set_title('EEG Seizure Detection - Model Performance Comparison',
                  fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(index + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.10)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=11)  # ✅ Removed framestyle
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(y=0.95, color='green', linestyle=':', alpha=0.6, linewidth=2, label='95% Threshold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()