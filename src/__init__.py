"""
EEG Seizure Detection Package
==============================
Deep learning pipeline for automated seizure detection from EEG signals.

Version: 1.0.0
Author: Your Name
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your@email.com"

from .preprocessing import filters
from .models import alexnet, densenet, googlenet, vgg, resnet, rnn
from .training import trainer
from .utils import visualization

__all__ = [
    "filters",
    "alexnet",
    "densenet", 
    "googlenet",
    "vgg",
    "resnet",
    "rnn",
    "trainer",
    "visualization",
]
