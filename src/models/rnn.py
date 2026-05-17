"""
Simple Dense Baseline (RNN-style) for EEG Seizure Classification
=================================================================
A lightweight Flatten → Dense → Sigmoid baseline used as a
performance lower-bound comparison against deeper architectures.

Input  : flat vector of 178 EEG time points
Output : sigmoid probability (binary classification)
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout


def build_eeg_rnn(input_dim: int = 178) -> Sequential:
    """Build a simple dense baseline model."""
    model = Sequential(
        [
            Flatten(input_shape=(input_dim,), name="flatten"),
            Dense(64, activation="relu", name="fc1"),
            Dropout(0.3, name="dropout"),
            Dense(1, activation="sigmoid", name="output"),
        ],
        name="EEG_RNN_Baseline",
    )
    return model


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("Testing RNN Baseline for EEG Seizure Detection")
    print("=" * 60)
    
    model = build_eeg_rnn(input_dim=178)
    model.summary()
    
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 178).astype("float32")
    output = model(dummy_input, training=False)
    
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("\n✅ RNN model test passed!")
