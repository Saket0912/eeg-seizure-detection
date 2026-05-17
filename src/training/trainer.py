"""
Model Trainer
=============
Handles compilation, training, evaluation, and EWC fine-tuning
for all EEG seizure detection models.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Dict, List, Any, Optional


class Trainer:
    """Unified trainer for EEG seizure detection models."""

    def __init__(
        self,
        model: tf.keras.Model,
        loss: str = "sparse_categorical_crossentropy",
        optimizer: str = "adam",
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.history = None

    def compile(self, **kwargs) -> None:
        """Compile the model with stored loss and optimizer."""
        self.model.compile(
            optimizer=kwargs.get("optimizer", self.optimizer),
            loss=kwargs.get("loss", self.loss),
            metrics=kwargs.get("metrics", ["accuracy"]),
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 128,
        compile_model: bool = True,
    ) -> tf.keras.callbacks.History:
        """Compile and train the model."""
        if compile_model:
            self.compile()

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
        )
        return self.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        binary: bool = False,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """Evaluate the model and return a metrics dictionary."""
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)

        y_pred_raw = self.model.predict(X_test, verbose=0)

        if binary:
            y_pred = (y_pred_raw > 0.5).astype("int32").ravel()
        else:
            y_pred = np.argmax(y_pred_raw, axis=1)

        y_true = y_test.ravel()

        metrics = {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "precision": float(precision_score(y_true, y_pred, average=average,
                                         zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average,
                                   zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }

        print(f"\n{'='*45}")
        print(f"  Model   : {self.model.name}")
        print(f"  Loss    : {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"{'='*45}")
        print(classification_report(y_true, y_pred,
                                    target_names=["No Seizure", "Seizure"]))
        return metrics

    def ewc_finetune(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        X_val_new: np.ndarray,
        y_val_new: np.ndarray,
        fisher_information: List[tf.Tensor],
        prev_params: List[tf.Variable],
        lambda_ewc: float = 0.1,
        epochs: int = 5,
        batch_size: int = 128,
    ) -> tf.keras.callbacks.History:
        """Fine-tune the model on a new task using EWC regularisation."""
        ewc_penalty = 0.5 * lambda_ewc * sum(
            tf.reduce_sum(f * tf.square(p - p_prev))
            for f, p, p_prev in zip(
                fisher_information, self.model.trainable_variables, prev_params
            )
        )

        def ewc_loss_fn(y_true, y_pred):
            base = tf.keras.losses.sparse_categorical_crossentropy(
                y_true, y_pred
            )
            return base + ewc_penalty

        self.model.compile(
            optimizer="adam",
            loss=ewc_loss_fn,
            metrics=["accuracy"],
        )

        self.history = self.model.fit(
            X_new,
            y_new,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_new, y_val_new),
            verbose=1,
        )
        return self.history
