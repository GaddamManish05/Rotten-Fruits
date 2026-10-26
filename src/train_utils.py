"""
FreshGuard AI - Training Utilities
Provides standard training execution, callbacks, metrics persistence, and plotting.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf

import config


def get_standard_callbacks(
    model_save_path: Path,
    patience_early_stopping: int = config.PATIENCE_EARLY_STOPPING,
    patience_reduce_lr: int = config.PATIENCE_REDUCE_LR,
) -> list:
    """Returns standardized Keras callbacks for reliable, reproducible training."""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=patience_reduce_lr,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks


def save_training_history(history: tf.keras.callbacks.History, duration: float, save_path: Path) -> Dict:
    """Extracts history dict, adds metadata, and saves as JSON."""
    history_dict = {}
    for k, v in history.history.items():
        # Convert float32 to python floats for JSON serialization
        history_dict[k] = [float(val) for val in v]
    history_dict["duration_seconds"] = round(duration, 2)
    history_dict["total_epochs_trained"] = len(history_dict.get("loss", []))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    return history_dict


def plot_and_save_curves(history_data: Dict, model_name: str, output_path: Path):
    """
    Plots training vs validation accuracy and loss curves and saves to output_path.
    """
    epochs = range(1, len(history_data.get("loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    if "accuracy" in history_data:
        ax1.plot(epochs, history_data["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history_data:
        ax1.plot(epochs, history_data["val_accuracy"], "g--s", label="Validation Accuracy", linewidth=2)
    ax1.set_title(f"{model_name} - Accuracy Curve", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim([0, 1.02])
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    if "loss" in history_data:
        ax2.plot(epochs, history_data["loss"], "r-o", label="Training Loss", linewidth=2)
    if "val_loss" in history_data:
        ax2.plot(epochs, history_data["val_loss"], "m--s", label="Validation Loss", linewidth=2)
    ax2.set_title(f"{model_name} - Loss Curve", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    model_save_path: Path,
    history_save_path: Path,
    plot_save_path: Path,
    class_weights: Optional[Dict[int, float]] = None,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Executes complete training loop with timing, callbacks, history persistence, and plots.
    """
    callbacks = get_standard_callbacks(model_save_path)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    duration = time.time() - start_time

    # Save trained model explicitly if early stopping didn't already
    model.save(str(model_save_path))

    history_dict = save_training_history(history, duration, history_save_path)
    plot_and_save_curves(history_dict, model.name, plot_save_path)

    print(f"\nTraining completed for {model.name} in {duration:.1f}s.")
    print(f"Model saved to: {model_save_path}")
    print(f"History saved to: {history_save_path}")
    print(f"Curves saved to: {plot_save_path}\n")

    return model, history_dict
