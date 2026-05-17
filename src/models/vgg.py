"""
1D VGG-16 for EEG Seizure Classification
==========================================
Adapted from Simonyan & Zisserman (2014) for 1-D EEG time-series.

Input shape : (178, 1)
Output      : softmax over num_classes
"""

from tensorflow.keras import layers, models
from typing import Tuple


def build_eeg_vgg(
    input_shape: Tuple[int, int] = (178, 1),
    num_classes: int = 2,
) -> models.Sequential:
    """Build a 1D VGG-16 model for EEG classification."""
    model = models.Sequential(name="EEG_VGG16")

    def conv_block(filters: int, n_convs: int, first: bool = False) -> None:
        for i in range(n_convs):
            kwargs = {"padding": "same", "activation": "relu"}
            if first and i == 0:
                kwargs["input_shape"] = input_shape
            model.add(layers.Conv1D(filters, 3, **kwargs))
        model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    conv_block(64,  n_convs=2, first=True)
    conv_block(128, n_convs=2)
    conv_block(256, n_convs=3)
    conv_block(512, n_convs=3)
    conv_block(512, n_convs=3)

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    return model


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("Testing VGG for EEG Seizure Detection")
    print("=" * 60)
    
    model = build_eeg_vgg(input_shape=(178, 1), num_classes=2)
    model.summary()
    
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 178, 1).astype("float32")
    output = model(dummy_input, training=False)
    
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("\n✅ VGG model test passed!")
