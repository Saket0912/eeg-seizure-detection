"""
1D AlexNet for EEG Seizure Classification
==========================================
Adapted from Krizhevsky et al. (2012) for 1-D EEG time-series.

Architecture
------------
- Conv1D layers replace Conv2D
- Input shape: (178, 1) — 178 time points, 1 channel
- Output: Softmax over 2 classes (seizure / no seizure)

References
----------
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
ImageNet classification with deep convolutional neural networks.
In NeurIPS.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


def build_eeg_alexnet(
    input_shape: Tuple[int, int] = (178, 1),
    num_classes: int = 2,
) -> models.Sequential:
    """
    Build a 1D AlexNet model for EEG seizure classification.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input sample, e.g. ``(178, 1)``.
    num_classes : int
        Number of output classes (2 for binary seizure detection).

    Returns
    -------
    tf.keras.Sequential
        Uncompiled Keras model.
    """
    model = models.Sequential(name="EEG_AlexNet")

    # Block 1
    model.add(
        layers.Conv1D(
            filters=96,
            kernel_size=11,
            strides=4,
            activation="relu",
            input_shape=input_shape,
            padding="valid",
            name="conv1"
        )
    )
    model.add(layers.MaxPooling1D(pool_size=3, strides=2, name="pool1"))

    # Block 2
    model.add(
        layers.Conv1D(
            filters=256,
            kernel_size=5,
            padding="same",
            activation="relu",
            name="conv2"
        )
    )
    model.add(layers.MaxPooling1D(pool_size=3, strides=2, name="pool2"))

    # Block 3
    model.add(
        layers.Conv1D(
            filters=384,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="conv3"
        )
    )

    # Block 4
    model.add(
        layers.Conv1D(
            filters=384,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="conv4"
        )
    )

    # Block 5
    model.add(
        layers.Conv1D(
            filters=256,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="conv5"
        )
    )
    model.add(layers.MaxPooling1D(pool_size=3, strides=2, name="pool3"))

    # Classification Layers
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(4096, activation="relu", name="fc1"))
    model.add(layers.Dropout(0.5, name="dropout1"))
    model.add(layers.Dense(4096, activation="relu", name="fc2"))
    model.add(layers.Dropout(0.5, name="dropout2"))
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    return model


def compile_alexnet(
    model: models.Sequential,
    learning_rate: float = 0.001,
) -> models.Sequential:
    """Compile the AlexNet model with standard settings."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("Testing AlexNet for EEG Seizure Detection")
    print("=" * 60)
    
    alexnet = build_eeg_alexnet(input_shape=(178, 1), num_classes=2)
    alexnet.summary()
    
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 178, 1).astype("float32")
    output = alexnet(dummy_input, training=False)
    
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("\n✅ AlexNet model test passed!")
