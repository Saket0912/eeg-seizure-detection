#!/usr/bin/env python3
"""
EEG Seizure Detection — Main Entry Point
=========================================
Runs the complete pipeline:
  1. Load & preprocess data
  2. Apply Butterworth + Wavelet filter (best SNR)
  3. Train all six models
  4. Evaluate and compare performance
  5. EWC fine-tuning of GoogLeNet on Dataset 2
  6. Save all result plots to results/

Usage
-----
    python main.py
    python main.py --model googlenet --epochs 10
    python main.py --preprocess_only
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.models.alexnet import build_eeg_alexnet
from src.models.densenet import build_eeg_densenet
from src.models.googlenet import build_eeg_googlenet
from src.models.resnet import build_eeg_resnet
from src.models.rnn import build_eeg_rnn
from src.models.vgg import build_eeg_vgg
from src.preprocessing.filters import (
    apply_butterworth_wavelet_filter,
    compare_all_filters,
)
from src.training.trainer import Trainer
from src.utils.visualization import (
    plot_filtered_eeg,
    plot_model_comparison,
    plot_raw_eeg,
    plot_snr_comparison,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEG Seizure Detection Pipeline"
    )
    parser.add_argument("--data_dir",       default="data/",     type=str)
    parser.add_argument("--output_dir",     default="results/",  type=str)
    parser.add_argument("--epochs",         default=10,          type=int)
    parser.add_argument("--batch_size",     default=128,         type=int)
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "alexnet", "densenet", "googlenet",
                 "vgg", "resnet", "rnn"],
    )
    parser.add_argument("--preprocess_only", action="store_true")
    return parser.parse_args()


def load_and_preprocess(data_dir: str):
    """Load CSVs, clean, filter, and split into (X, y) pairs."""
    print("\n[1/5] Loading datasets …")
    
    df1_path = os.path.join(data_dir, "EEG_Dataset-1.csv")
    df2_path = os.path.join(data_dir, "EEG_Dataset-2.csv")
    
    if not os.path.exists(df1_path) or not os.path.exists(df2_path):
        print(f"[!] Warning: Dataset files not found in {data_dir}")
        print("[!] Creating synthetic data for demonstration...")
        
        np.random.seed(42)
        n_samples = 1000
        n_features = 178
        
        X1 = np.random.randn(n_samples, n_features)
        y1 = np.random.randint(0, 2, n_samples)
        
        X2 = np.random.randn(n_samples, n_features)
        y2 = np.random.randint(0, 2, n_samples)
        
        return X1, X2, y1, y2, None, X1
    
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    for df in (df1, df2):
        df.dropna(inplace=True)

    df1 = df1.iloc[:, 1:]
    df2 = df2.iloc[:, 1:]

    X1_raw = df1.iloc[:, :-1].values.astype(float)
    X2_raw = df2.iloc[:, :-1].values.astype(float)
    y1 = df1.iloc[:, -1].values
    y2 = df2.iloc[:, -1].values

    print("[2/5] Comparing filter SNRs …")
    compare_all_filters(X1_raw)

    print("[3/5] Applying Butterworth + Wavelet filter …")
    X1_filt = apply_butterworth_wavelet_filter(X1_raw)
    X2_filt = apply_butterworth_wavelet_filter(X2_raw)

    y1 = np.where(y1 > 1, 0, y1)
    y2 = np.where(y2 > 1, 0, y2)

    return X1_filt, X2_filt, y1, y2, df1, X1_raw


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X1, X2, y1, y2, df1_raw, X1_raw = load_and_preprocess(args.data_dir)

    if df1_raw is not None:
        df1_raw_df = pd.DataFrame(
            X1_raw, columns=[f"X{i}" for i in range(1, X1_raw.shape[1] + 1)]
        )
        filtered_df = pd.DataFrame(
            X1, columns=[f"X{i}" for i in range(1, X1.shape[1] + 1)]
        )
        plot_raw_eeg(df1_raw_df, title="Raw EEG Signals",
                     save_path=f"{args.output_dir}/raw_eeg.png")
        plot_filtered_eeg(filtered_df, title="Filtered EEG Signals",
                          save_path=f"{args.output_dir}/filtered_eeg.png")

    if args.preprocess_only:
        print("Preprocessing complete. Exiting.")
        return

    print("\n[4/5] Training models …\n")

    from sklearn.model_selection import train_test_split
    
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X1, y1, test_size=0.2, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42
    )

    X_tr3  = np.expand_dims(X_tr,  -1)
    X_val3 = np.expand_dims(X_val, -1)
    X_te3  = np.expand_dims(X_te,  -1)

    results = {}

    if args.model in ("all", "alexnet"):
        print("  Training AlexNet...")
        m = build_eeg_alexnet()
        t = Trainer(m)
        h = t.train(X_tr3, y_tr, X_val3, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)
        results["AlexNet"] = t.evaluate(X_te3, y_te)
        plot_training_curves(h, "AlexNet",
                             save_path=f"{args.output_dir}/alexnet_curves.png")

    if args.model in ("all", "densenet"):
        print("  Training DenseNet...")
        m = build_eeg_densenet()
        t = Trainer(m, loss="binary_crossentropy")
        h = t.train(X_tr, y_tr, X_val, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)
        results["DenseNet"] = t.evaluate(X_te, y_te, binary=True)
        plot_training_curves(h, "DenseNet",
                             save_path=f"{args.output_dir}/densenet_curves.png")

    if args.model in ("all", "googlenet"):
        print("  Training GoogLeNet...")
        m = build_eeg_googlenet()
        t = Trainer(m)
        h = t.train(X_tr3, y_tr, X_val3, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)
        results["GoogLeNet"] = t.evaluate(X_te3, y_te)
        plot_training_curves(h, "GoogLeNet",
                             save_path=f"{args.output_dir}/googlenet_curves.png")

    if args.model in ("all", "vgg"):
        print("  Training VGG...")
        m = build_eeg_vgg()
        t = Trainer(m)
        h = t.train(X_tr3, y_tr, X_val3, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)
        results["VGG"] = t.evaluate(X_te3, y_te)
        plot_training_curves(h, "VGG",
                             save_path=f"{args.output_dir}/vgg_curves.png")

    if args.model in ("all", "resnet"):
        print("  Training ResNet...")
        m = build_eeg_resnet()
        t = Trainer(m)
        h = t.train(X_tr3, y_tr, X_val3, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)
        results["ResNet"] = t.evaluate(X_te3, y_te)
        plot_training_curves(h, "ResNet",
                             save_path=f"{args.output_dir}/resnet_curves.png")

    if args.model in ("all", "rnn"):
        print("  Training RNN Baseline...")
        m = build_eeg_rnn()
        t = Trainer(m, loss="binary_crossentropy")
        h = t.train(X_tr, y_tr, X_val, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)
        results["RNN"] = t.evaluate(X_te, y_te, binary=True)
        plot_training_curves(h, "RNN",
                             save_path=f"{args.output_dir}/rnn_curves.png")

    if results:
        print("\n[5/5] Generating comparison chart …")
        metrics_chart = {
            "Accuracy":  [v["accuracy"]  for v in results.values()],
            "Precision": [v["precision"] for v in results.values()],
            "Recall":    [v["recall"]    for v in results.values()],
            "F1-Score":  [v["f1"]        for v in results.values()],
        }
        plot_model_comparison(
            model_names=list(results.keys()),
            metrics=metrics_chart,
            save_path=f"{args.output_dir}/model_comparison.png",
        )
        print(f"\n✅ All results saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
