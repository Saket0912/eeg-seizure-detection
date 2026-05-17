"""Unit tests for model architectures — shape checks only (no training)."""

import numpy as np
import pytest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.models.alexnet import build_eeg_alexnet
from src.models.densenet import build_eeg_densenet
from src.models.googlenet import build_eeg_googlenet
from src.models.resnet import build_eeg_resnet
from src.models.rnn import build_eeg_rnn
from src.models.vgg import build_eeg_vgg

BATCH = 4


class TestAlexNet:
    def test_output_shape(self):
        m = build_eeg_alexnet(input_shape=(178, 1), num_classes=2)
        x = np.random.randn(BATCH, 178, 1).astype("float32")
        y = m(x, training=False)
        assert y.shape == (BATCH, 2)

    def test_layer_count(self):
        m = build_eeg_alexnet()
        assert len(m.layers) > 5


class TestDenseNet:
    def test_output_shape(self):
        m = build_eeg_densenet(input_dim=178)
        x = np.random.randn(BATCH, 178).astype("float32")
        y = m(x, training=False)
        assert y.shape == (BATCH, 1)

    def test_sigmoid_range(self):
        m = build_eeg_densenet()
        x = np.random.randn(BATCH, 178).astype("float32")
        y = m(x, training=False).numpy()
        assert np.all(y >= 0) and np.all(y <= 1)


class TestGoogLeNet:
    def test_output_shape(self):
        m = build_eeg_googlenet(input_shape=(178, 1), num_classes=2)
        x = np.random.randn(BATCH, 178, 1).astype("float32")
        y = m(x, training=False)
        assert y.shape == (BATCH, 2)

    def test_softmax_sums_to_one(self):
        m = build_eeg_googlenet()
        x = np.random.randn(BATCH, 178, 1).astype("float32")
        y = m(x, training=False).numpy()
        np.testing.assert_allclose(y.sum(axis=1), np.ones(BATCH), atol=1e-5)


class TestVGG:
    def test_output_shape(self):
        m = build_eeg_vgg(input_shape=(178, 1), num_classes=2)
        x = np.random.randn(BATCH, 178, 1).astype("float32")
        y = m(x, training=False)
        assert y.shape == (BATCH, 2)


class TestResNet:
    def test_output_shape(self):
        m = build_eeg_resnet(input_shape=(178, 1), num_classes=2)
        x = np.random.randn(BATCH, 178, 1).astype("float32")
        y = m(x, training=False)
        assert y.shape == (BATCH, 2)


class TestRNN:
    def test_output_shape(self):
        m = build_eeg_rnn(input_dim=178)
        x = np.random.randn(BATCH, 178).astype("float32")
        y = m(x, training=False)
        assert y.shape == (BATCH, 1)
