"""
1D ResNet for EEG Seizure Classification
=========================================
Adapted from He et al. (2016) for 1-D EEG time-series.

Input shape : (178, 1)
Output      : softmax over num_classes
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple


def _residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
) -> tf.Tensor:
    """Single residual block with optional shortcut projection."""
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, strides=strides,
                       padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if strides != 1:
        shortcut = layers.Conv1D(filters, kernel_size=1,
                                  strides=strides, padding="same")(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_eeg_resnet(
    input_shape: Tuple[int, int] = (178, 1),
    num_classes: int = 2,
) -> tf.keras.Model:
    """Build a 1D ResNet model for EEG classification."""
    inputs = tf.keras.Input(shape=input_shape, name="eeg_input")

    x = layers.Conv1D(64, 7, strides=2, padding="same",
                       name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same",
                             name="stem_pool")(x)

    x = _residual_block(x, filters=64, kernel_size=3)
    x = _residual_block(x, filters=64, kernel_size=3)
    x = _residual_block(x, filters=64, kernel_size=3)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                            name="output")(x)

    return tf.keras.Model(inputs, outputs, name="EEG_ResNet")


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("Testing ResNet for EEG Seizure Detection")
    print("=" * 60)
    
    model = build_eeg_resnet(input_shape=(178, 1), num_classes=2)
    model.summary()
    
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 178, 1).astype("float32")
    output = model(dummy_input, training=False)
    
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("\n✅ ResNet model test passed!")
