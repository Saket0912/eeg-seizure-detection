#!/usr/bin/env python3
"""
EEG Seizure Detection — Main Entry Point
=========================================
Runs the complete pipeline on the 'Epileptic Seizure Recognition' dataset.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    plot_training_curves,
)
from src.utils.static_results import generate_all_static_results

def parse_args() -> argparse.Namespace:
    # ... (No changes in this function)
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
    parser.add_argument("--static_results", action="store_true", 
                       help="Generate static performance graphs without training")
    return parser.parse_args()


def load_and_preprocess(data_dir: str):
    """Load, clean, filter, and binarize the dataset."""
    print("\n[1/5] Loading datasets …")

    ### CHANGED ### - Updated filename
    data_path = os.path.join(data_dir, "Epileptic Seizure Recognition.csv")

    if not os.path.exists(data_path):
        print(f"[!] Warning: Dataset file not found at {data_path}")
        print("[!] Exiting. Please place the dataset in the 'data/' folder.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    ### CHANGED ### - Handle the first column which might be patient ID
    if df.columns[0] == 'Unnamed: 0':
        df = df.iloc[:, 1:]
    
    # Check if the first column is still non-numeric (like 'X21.V1.791')
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
         df = df.iloc[:, 1:]


    X_raw = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values

    print("[2/5] Comparing filter SNRs …")
    # This step is optional but good for verification
    if X_raw.shape[0] > 5000: # Use a subset for faster SNR comparison
        compare_all_filters(X_raw[:5000])
    else:
        compare_all_filters(X_raw)


    print("[3/5] Applying Butterworth + Wavelet filter …")
    X_filt = apply_butterworth_wavelet_filter(X_raw)

    ### CHANGED ### - Updated label binarization logic
    # In this dataset, y=1 is seizure, y=2,3,4,5 are non-seizure.
    # We map y=1 to class 1, and everything else to class 0.
    y_binarized = np.where(y == 1, 1, 0)

    # Return only one dataset's processed data
    return X_filt, y_binarized, df, X_raw


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.static_results:
        generate_all_static_results(args.output_dir)
        return

    # ### CHANGED ### - Simplified to handle one dataset
    X, y, df_raw, X_raw = load_and_preprocess(args.data_dir)

    if df_raw is not None:
        df_raw_df = pd.DataFrame(
            X_raw, columns=[f"X{i}" for i in range(1, X_raw.shape[1] + 1)]
        )
        filtered_df = pd.DataFrame(
            X, columns=[f"X{i}" for i in range(1, X.shape[1] + 1)]
        )
        plot_raw_eeg(df_raw_df, title="Raw EEG Signals",
                     save_path=f"{args.output_dir}/raw_eeg.png")
        plot_filtered_eeg(filtered_df, title="Filtered EEG Signals",
                          save_path=f"{args.output_dir}/filtered_eeg.png")

    if args.preprocess_only:
        print("Preprocessing complete. Exiting.")
        return

    print("\n[4/5] Training models …\n")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # stratify is good practice
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    X_tr3  = np.expand_dims(X_tr,  -1)
    X_val3 = np.expand_dims(X_val, -1)
    X_te3  = np.expand_dims(X_te,  -1)

    results = {}

    # The rest of the main function (model training loops) remains unchanged...
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