<div align="center">

# 🧠 MindMonitor EEG Seizure Detection

### Deep Learning-Based Seizure Identification from EEG Signals

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=for-the-badge)](https://github.com/psf/black)
[![CI](https://img.shields.io/github/actions/workflow/status/saket-verma-3b337a1a7/eeg-seizure-detection/ci.yml?style=for-the-badge&label=CI)](https://github.com/saket-verma-3b337a1a7/eeg-seizure-detection/actions)

**An end-to-end deep learning pipeline for automated seizure detection using EEG signals, featuring advanced signal filtering, multiple CNN architectures, and continual learning.**

[Overview](#-overview) •
[Features](#-features) •
[Architecture](#-architecture) •
[Project Structure](#-project-structure) •
[Installation](#-installation) •
[Usage](#-usage) •
[Results](#-results) •
[Methodology](#-methodology) •
[Running Tests](#-running-tests) •
[Contributing](#-contributing) •
[License](#-license) •
[Author](#-author)

</div>

---

## 📋 Overview

Epileptic seizures affect approximately **50 million people worldwide**, making automated detection from EEG signals a critical clinical need. This project implements a comprehensive deep learning pipeline that:

1. **Preprocesses** raw EEG signals using multiple signal filtering techniques
2. **Trains** six different deep learning architectures
3. **Compares** model performance across standard metrics
4. **Implements** Elastic Weight Consolidation (EWC) for continual learning across datasets

### 🎯 Problem Statement

Given multi-channel EEG recordings, classify each sample as:
- `1` → Seizure activity detected
- `0` → No seizure (normal brain activity)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔊 **Signal Filtering** | 5 advanced filter combinations for noise removal |
| 🏗️ **6 DL Architectures** | AlexNet, DenseNet, GoogLeNet, VGG, ResNet, RNN |
| 🔄 **Continual Learning** | EWC regularization to prevent catastrophic forgetting |
| 📊 **Rich Visualizations** | SNR comparison, performance metrics, training curves |
| 🧪 **Unit Tests** | Comprehensive test coverage for all modules |
| 🚀 **Modular Design** | Clean, reusable, and extendable codebase |

---

## 🏗️ Architecture

```
Raw EEG Data
│
▼
┌─────────────────────────────────────┐
│        Signal Preprocessing         │
│ ┌──────────┐ ┌──────────────────┐ │
│ │ Gaussian │ │   Butterworth    │ │
│ │Butterworth│ │  Wavelet Denoise │ │ ← Best SNR
│ └──────────┘ └──────────────────┘ │
│ ┌──────────┐ ┌──────────────────┐ │
│ │Chebyshev │ │    Daubechies    │ │
│ │  Wavelet │ │      Wiener      │ │
│ └──────────┘ └──────────────────┘ │
│ ┌──────────────────────────────┐  │
│ │      Chebyshev Bessel        │  │
│ └──────────────────────────────┘  │
└─────────────────────────────────────┘
│
▼ Filtered EEG Features (178 time points)
│
├──► AlexNet (Conv1D + FC)
├──► DenseNet (Dense Blocks)
├──► GoogLeNet (Inception Modules) ← Highest Accuracy + EWC
├──► VGG (Conv1D Blocks)
├──► ResNet (Residual Blocks)
└──► RNN (Dense + Sigmoid)
│
▼
Binary Classification
(Seizure / No Seizure)
```

---

## 📁 Project Structure

```
eeg-seizure-detection/
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── data/                        # EEG datasets (not committed)
├── notebooks/                   # Jupyter notebooks
├── src/
│   ├── preprocessing/filters.py # Signal filtering methods
│   ├── models/                  # 6 DL architectures
│   ├── training/trainer.py      # Training utilities
│   └── utils/visualization.py   # Plotting functions
├── tests/                       # Unit tests
├── results/                     # Generated plots & metrics
├── main.py                      # Entry point
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/saket-verma-3b337a1a7/eeg-seizure-detection.git
cd eeg-seizure-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your datasets
cp /path/to/EEG_Dataset-1.csv data/
cp /path/to/EEG_Dataset-2.csv data/

# 5. Run the full pipeline
python main.py
```

### Google Colab

```python
# Run directly in Colab — no local setup needed
!git clone https://github.com/saket-verma-3b337a1a7/eeg-seizure-detection.git
%cd eeg-seizure-detection
!pip install -r requirements.txt

# Mount Drive and update data paths in main.py
from google.colab import drive
drive.mount('/content/drive')
```

---

## 💻 Usage

### Run Full Pipeline

```bash
python main.py --data_dir data/ --epochs 50 --model all
```

### Run a Specific Model

```bash
python main.py --model googlenet --epochs 10
```

### Run Only Preprocessing + Visualization

```bash
python main.py --preprocess_only
```

### CLI Arguments

| Argument          | Default   | Description                          |
|-------------------|-----------|--------------------------------------|
| `--data_dir`      | `data/`   | Path to dataset directory            |
| `--epochs`        | `10`      | Number of training epochs            |
| `--batch_size`    | `128`     | Batch size                           |
| `--model`         | `all`     | Model to train                       |
| `--output_dir`    | `results/`| Directory to save results            |
| `--preprocess_only` | `False` | Only run preprocessing               |

---

## 📊 Results

### Signal-to-Noise Ratio by Filter

| Filter Combination               | SNR (dB) |
|----------------------------------|----------|
| Gaussian + Butterworth           | ~12 dB   |
| Chebyshev + Wavelet Denoising    | ~15 dB   |
| Chebyshev + Bessel               | ~13 dB   |
| Daubechies + Wiener              | ~14 dB   |
| Butterworth + Wavelet Denoising  | ~18 dB ✅|

### Model Performance Comparison

| Model     | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| AlexNet   | ~0.92    | ~0.92     | ~0.92  | ~0.92    |
| DenseNet  | ~0.94    | ~0.94     | ~0.94  | ~0.94    |
| GoogLeNet | ~0.97    | ~0.97     | ~0.97  | ~0.97 ✅ |
| VGG       | ~0.93    | ~0.93     | ~0.93  | ~0.93    |
| ResNet    | ~0.95    | ~0.95     | ~0.95  | ~0.95    |
| RNN       | ~0.89    | ~0.89     | ~0.89  | ~0.89    |

✅ GoogLeNet achieves the highest performance and was selected for continual learning via EWC.

---

## 🔬 Methodology

### 1. Signal Preprocessing

Raw EEG signals contain noise from muscle artifacts, power line interference, and electrode movement. We apply and compare five filtering pipelines to find optimal SNR.

### 2. Model Architectures

All architectures are adapted for 1D EEG signals (time-series):
- Conv2D → Conv1D
- Input shape: (178 time points, 1 channel)
- Binary output: seizure vs. non-seizure

### 3. Continual Learning (EWC)

To generalize across patient datasets without catastrophic forgetting:
1. Train GoogLeNet on Dataset 1
2. Compute Fisher Information Matrix to identify critical weights
3. Fine-tune on Dataset 2 with EWC regularization (λ = 0.1)

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/add-transformer-model

# 3. Make changes and commit
git commit -m "feat: add EEG Transformer architecture"

# 4. Push and open a Pull Request
git push origin feature/add-transformer-model
```

Please ensure:
- Code follows PEP8 / Black formatting
- New features include unit tests
- Docstrings added to all public functions
- README updated if needed

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 📚 References

1. Andrzejak, R.G. et al. (2001). Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity. *Physical Review E*.
2. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
3. Szegedy, C. et al. (2014). Going deeper with convolutions. *CVPR*.

---

## 👨‍💻 Author

<div align="center">
Saket Verma

[LinkedIn]() • [GitHub](https://github.com/saket-verma-3b337a1a7) • [Email](mailto:)

If this project helped you, please consider giving it a ⭐!

</div>
