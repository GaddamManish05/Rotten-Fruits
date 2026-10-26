"""
FreshGuard AI - Model Evaluation & Comparison Module
Evaluates all trained models on the held-out test set.
Computes Accuracy, Precision, Recall, Macro-F1, Confusion Matrix, Latency, and Model Size.
Persists comparison metrics to JSON and outputs formatted tables.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_preprocessing import load_dataset_from_directory
from src.dataset_validation import discover_classes


def get_file_size_mb(path: Path) -> float:
    """Returns size of a file in Megabytes."""
    if not path.exists():
        return 0.0
    return round(os.path.getsize(path) / (1024 * 1024), 2)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], model_name: str, output_path: Path):
    """Generates and saves a high-quality heatmap visualization of the confusion matrix."""
    clean_labels = [f"{config.CLASS_FRESHNESS.get(c, '')} {config.CLASS_FRUIT_TYPE.get(c, c)}".strip() for c in class_names]
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar=True,
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_single_model(
    model_path: Path,
    history_path: Path,
    test_ds: tf.data.Dataset,
    true_labels: np.ndarray,
    class_names: List[str],
    experiment_name: str,
) -> Optional[Dict]:
    """
    Evaluates a single saved model on the test dataset.
    Returns dictionary with all metrics and metadata.
    """
    if not model_path.exists():
        print(f"Skipping {experiment_name}: Model checkpoint not found at {model_path}")
        return None

    print(f"\nEvaluating: {experiment_name} ({model_path.name})...")
    custom_objects = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
    }
    try:
        model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    # Measure test inference latency
    start_time = time.time()
    predictions = model.predict(test_ds, verbose=0)
    test_duration = time.time() - start_time

    num_samples = len(true_labels)
    latency_per_sample_ms = round((test_duration / max(num_samples, 1)) * 1000, 2)

    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = float(np.mean(predicted_labels == true_labels))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        true_labels, predicted_labels, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot Confusion Matrix
    cm_slug = experiment_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    cm_path = config.FIGURES_DIR / f"cm_{cm_slug}.png"
    plot_confusion_matrix(cm, class_names, experiment_name, cm_path)

    # Load training history if available
    training_duration = 0.0
    total_epochs = 0
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                h = json.load(f)
                training_duration = h.get("duration_seconds", 0.0)
                total_epochs = h.get("total_epochs_trained", len(h.get("loss", [])))
        except Exception:
            pass

    metrics = {
        "experiment_name": experiment_name,
        "model_file": model_path.name,
        "model_size_mb": get_file_size_mb(model_path),
        "total_parameters": int(model.count_params()),
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "test_latency_ms_per_sample": latency_per_sample_ms,
        "training_duration_seconds": training_duration,
        "epochs_trained": total_epochs,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "cm_plot_path": str(cm_path.relative_to(config.BASE_DIR)),
    }

    return metrics


def evaluate_all():
    """Runs evaluation for all trained experiments and outputs comparison table."""
    print("=" * 60)
    print("FreshGuard AI - Comprehensive Model Evaluation")
    print("=" * 60)

    if not config.TEST_DIR.exists():
        print(f"ERROR: Test directory not found at {config.TEST_DIR}")
        return {}

    classes = discover_classes(config.TEST_DIR)
    if not classes:
        print("ERROR: No class subdirectories found in test directory.")
        return {}

    print(f"Loading held-out test set from: {config.TEST_DIR}")
    print(f"Classes ({len(classes)}): {classes}")

    test_ds, class_names = load_dataset_from_directory(
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        class_names=classes,
    )

    # Extract true ground-truth labels
    true_labels = []
    for _, y_batch in test_ds:
        true_labels.extend(np.argmax(y_batch.numpy(), axis=1))
    true_labels = np.array(true_labels)

    if len(true_labels) == 0:
        print("ERROR: Test set contains 0 samples.")
        return {}

    print(f"Loaded {len(true_labels)} test samples.")

    experiments = [
        ("Experiment A (Scratch CNN)", config.MODEL_SCRATCH_PATH, config.HISTORY_SCRATCH_PATH),
        ("Experiment B (MobileNetV2 Feature Extraction)", config.MODEL_FE_PATH, config.HISTORY_FE_PATH),
        ("Experiment C (MobileNetV2 Fine-Tuning)", config.MODEL_FT_PATH, config.HISTORY_FT_PATH),
    ]

    if config.MODEL_EFFICIENTNET_PATH.exists():
        experiments.append(
            ("Experiment D (EfficientNetB0)", config.MODEL_EFFICIENTNET_PATH, config.METRICS_DIR / "history_efficientnet.json")
        )

    results = {}
    for name, model_path, hist_path in experiments:
        metrics = evaluate_single_model(
            model_path=model_path,
            history_path=hist_path,
            test_ds=test_ds,
            true_labels=true_labels,
            class_names=class_names,
            experiment_name=name,
        )
        if metrics is not None:
            results[name] = metrics

    # Save comparison dictionary
    config.COMPARISON_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.COMPARISON_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY ON HELD-OUT TEST SET")
    print("=" * 80)
    header = f"{'Model Experiment':<45} | {'Accuracy':<9} | {'Macro F1':<9} | {'Latency':<9} | {'Size':<8}"
    print(header)
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<45} | {m['accuracy']*100:>7.2f}% | {m['macro_f1']:>9.4f} | {m['test_latency_ms_per_sample']:>6.1f} ms | {m['model_size_mb']:>5.1f} MB")
    print("=" * 80)
    print(f"Metrics saved to: {config.COMPARISON_METRICS_PATH}\n")

    return results


if __name__ == "__main__":
    evaluate_all()
