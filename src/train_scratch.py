"""
FreshGuard AI - Experiment A: Baseline Scratch CNN Training
Trains a convolutional neural network from scratch with random weight initialization.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_preprocessing import create_train_pipeline
from src.dataset_validation import get_dataset_summary, compute_class_weights
from src.model_factory import build_scratch_cnn, count_parameters
from src.train_utils import train_model


def run(epochs: int = config.DEFAULT_EPOCHS, batch_size: int = config.BATCH_SIZE):
    print("=" * 60)
    print("EXPERIMENT A: Training Scratch CNN Baseline")
    print("=" * 60)

    summary = get_dataset_summary()
    if not summary["is_ready_for_training"]:
        print("ERROR: Dataset is not prepared or empty. Run dataset preparation first.")
        sys.exit(1)

    class_names = summary["classes"]
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    train_ds, val_ds, _ = create_train_pipeline(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        batch_size=batch_size,
        apply_augmentation=True,
    )

    class_weights = compute_class_weights(summary["train"]["counts"])

    model = build_scratch_cnn(num_classes=num_classes)
    params = count_parameters(model)
    print(f"Model Architecture: Scratch CNN")
    print(f"Trainable Parameters: {params['trainable']:,} | Non-trainable: {params['non_trainable']:,}")

    plot_path = config.FIGURES_DIR / "curves_scratch.png"

    trained_model, history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        model_save_path=config.MODEL_SCRATCH_PATH,
        history_save_path=config.HISTORY_SCRATCH_PATH,
        plot_save_path=plot_path,
        class_weights=class_weights,
    )

    print("Experiment A completed successfully.")
    return trained_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scratch CNN Baseline")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size)
