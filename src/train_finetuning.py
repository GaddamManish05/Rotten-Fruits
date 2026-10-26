"""
FreshGuard AI - Experiment C: MobileNetV2 Fine-Tuning
Unfreezes top convolutional layers of MobileNetV2 while keeping early layers frozen.
Trained with a small learning rate (1e-5) to adapt features to fruit domain without catastrophic forgetting.
"""

import argparse
import sys
from pathlib import Path
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_preprocessing import create_train_pipeline
from src.dataset_validation import get_dataset_summary, compute_class_weights
from src.model_factory import build_mobilenetv2_finetuning, count_parameters
from src.train_utils import train_model


def run(epochs: int = 10, batch_size: int = config.BATCH_SIZE, unfreeze_from: int = 100):
    print("=" * 60)
    print("EXPERIMENT C: MobileNetV2 Fine-Tuning (Partial Unfreezing)")
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

    # Warm start from Feature Extraction model if available
    base_fe_model = None
    if config.MODEL_FE_PATH.exists():
        print(f"Loading warm-start weights from Feature Extraction model: {config.MODEL_FE_PATH}")
        try:
            base_fe_model = tf.keras.models.load_model(str(config.MODEL_FE_PATH))
        except Exception as e:
            print(f"Notice: Could not load FE weights directly ({e}). Initializing fresh model for fine-tuning.")
            base_fe_model = None

    model = build_mobilenetv2_finetuning(
        base_fe_model=base_fe_model,
        num_classes=num_classes,
        unfreeze_from_layer=unfreeze_from,
        learning_rate=config.LR_FINETUNING,
    )

    params = count_parameters(model)
    print(f"Model Architecture: MobileNetV2 (Fine-Tuning)")
    print(f"Trainable Parameters (Unfrozen top + Head): {params['trainable']:,} | Frozen Base: {params['non_trainable']:,}")

    plot_path = config.FIGURES_DIR / "curves_finetuned.png"

    trained_model, history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        model_save_path=config.MODEL_FT_PATH,
        history_save_path=config.HISTORY_FT_PATH,
        plot_save_path=plot_path,
        class_weights=class_weights,
    )

    print("Experiment C completed successfully.")
    return trained_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--unfreeze-from", type=int, default=100, help="Layer index to unfreeze from")
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, unfreeze_from=args.unfreeze_from)
