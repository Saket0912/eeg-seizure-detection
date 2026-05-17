"""
EEG Seizure Detection Models
=============================
All deep learning architectures for binary seizure classification.
"""

from .alexnet import build_eeg_alexnet
from .densenet import build_eeg_densenet
from .googlenet import build_eeg_googlenet
from .resnet import build_eeg_resnet
from .rnn import build_eeg_rnn
from .vgg import build_eeg_vgg

__all__ = [
    "build_eeg_alexnet",
    "build_eeg_densenet",
    "build_eeg_googlenet",
    "build_eeg_resnet",
    "build_eeg_rnn",
    "build_eeg_vgg",
]
