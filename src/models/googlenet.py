"""
1D GoogLeNet (Inception v1) for EEG Seizure Classification
===========================================================
Adapted from Szegedy et al. (2014) for 1-D EEG time-series.
Includes EWC (Elastic Weight Consolidation) support for
continual learning across patient datasets.

Input shape : (178, 1)
Output      : softmax over num_classes
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Tuple


def inception_module(x: tf.Tensor, filters: List[int]) -> tf.Tensor:
    """Single Inception module with four parallel branches."""
    # Branch 1 — 1×1 conv
    branch1 = layers.Conv1D(filters[0], 1, padding="same",
                             activation="relu")(x)

    # Branch 2 — 1×1 reduce → 3×3 conv
    branch2 = layers.Conv1D(filters[1], 1, padding="same",
                             activation="relu")(x)
    branch2 = layers.Conv1D(filters[2], 3, padding="same",
                             activation="relu")(branch2)

    # Branch 3 — 1×1 reduce → 5×5 conv
    branch3 = layers.Conv1D(filters[3], 1, padding="same",
                             activation="relu")(x)
    branch3 = layers.Conv1D(filters[4], 5, padding="same",
                             activation="relu")(branch3)

    # Branch 4 — MaxPool → 1×1 proj
    branch4 = layers.MaxPooling1D(3, strides=1, padding="same")(x)
    branch4 = layers.Conv1D(filters[5], 1, padding="same",
                             activation="relu")(branch4)

    return layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)


def build_eeg_googlenet(
    input_shape: Tuple[int, int] = (178, 1),
    num_classes: int = 2,
) -> tf.keras.Model:
    """Build a 1D GoogLeNet model for EEG classification."""
    inputs = tf.keras.Input(shape=input_shape, name="eeg_input")

    # Stem
    x = layers.Conv1D(64, 7, strides=2, padding="same",
                       activation="relu", name="stem_conv1")(inputs)
    x = layers.MaxPooling1D(3, strides=2, padding="same",
                             name="stem_pool1")(x)
    x = layers.Conv1D(64, 1, padding="same",
                       activation="relu", name="stem_conv2")(x)
    x = layers.Conv1D(192, 3, padding="same",
                       activation="relu", name="stem_conv3")(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same",
                             name="stem_pool2")(x)

    # Inception modules
    x = inception_module(x, filters=[64, 96, 128, 16, 32, 32])
    x = inception_module(x, filters=[128, 128, 192, 32, 96, 64])
    x = layers.MaxPooling1D(3, strides=2, padding="same",
                             name="inc_pool")(x)

    # Classifier
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(1024, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(1024, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.4, name="drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                            name="output")(x)

    return tf.keras.Model(inputs, outputs, name="EEG_GoogLeNet")


def compute_fisher_information(
    model: tf.keras.Model,
    X: tf.Tensor,
    y: tf.Tensor,
) -> List[tf.Tensor]:
    """Compute the diagonal Fisher Information Matrix (FIM) for each parameter."""
    with tf.GradientTape() as tape:
        predictions = model(X, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return [tf.square(g) for g in gradients]


def ewc_loss(
    model: tf.keras.Model,
    fisher_information: List[tf.Tensor],
    prev_params: List[tf.Variable],
    lambda_ewc: float = 0.1,
) -> tf.Tensor:
    """Compute the EWC regularisation penalty."""
    penalty = sum(
        tf.reduce_sum(f * tf.square(p - p_prev))
        for f, p, p_prev in zip(
            fisher_information, model.trainable_variables, prev_params
        )
    )
    return lambda_ewc * 0.5 * penalty


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("Testing GoogLeNet for EEG Seizure Detection")
    print("=" * 60)
    
    model = build_eeg_googlenet(input_shape=(178, 1), num_classes=2)
    model.summary()
    
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 178, 1).astype("float32")
    output = model(dummy_input, training=False)
    
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("\n✅ GoogLeNet model test passed!")
