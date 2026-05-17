"""
DenseNet for EEG Seizure Classification
=========================================
Custom dense-block architecture using fully connected layers
with LeakyReLU activations and dense connectivity.

Input  : flat vector of 178 EEG time points
Output : sigmoid (binary cross-entropy)
"""

import tensorflow as tf
from typing import Tuple


def _dense_block(inp: tf.Tensor, dims: int) -> tf.Tensor:
    """A single dense block: BN → 3 × Dense(LeakyReLU) → Dense(178)."""
    x = tf.keras.layers.BatchNormalization()(inp)
    for _ in range(3):
        x = tf.keras.layers.Dense(
            dims, activation=tf.keras.layers.LeakyReLU(0.2)
        )(x)
        x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        178, activation=tf.keras.layers.LeakyReLU(0.2)
    )(x)
    return x


def build_eeg_densenet(input_dim: int = 178) -> tf.keras.Model:
    """
    Build the DenseNet model for EEG binary classification.

    Three parallel dense blocks (dims=256 / 512 / 1024) are
    concatenated, reduced to 128 units, then output via sigmoid.
    """
    inp = tf.keras.layers.Input(shape=(input_dim,), name="eeg_input")

    x1 = _dense_block(inp, dims=256)
    x2 = _dense_block(inp, dims=512)
    x3 = _dense_block(inp, dims=1024)

    x = tf.keras.layers.Concatenate()([x1, x2, x3])
    x = tf.keras.layers.Dense(
        128, activation=tf.keras.layers.LeakyReLU(0.2), name="fc"
    )(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return tf.keras.Model(inp, out, name="EEG_DenseNet")


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("Testing DenseNet for EEG Seizure Detection")
    print("=" * 60)
    
    model = build_eeg_densenet()
    model.summary()
    
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 178).astype("float32")
    output = model(dummy_input, training=False)
    
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("\n✅ DenseNet model test passed!")
