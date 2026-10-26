"""
FreshGuard AI - Master Experiment Execution Pipeline
Runs all 3 core transfer learning experiments (Scratch, Feature Extraction, Fine-Tuning)
and automatically computes comparative test metrics, confusion matrices, and plots.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.train_scratch import run as run_scratch
from src.train_feature_extraction import run as run_feature_extraction
from src.train_finetuning import run as run_finetuning
from src.evaluate import evaluate_all


def run_pipeline(
    epochs_scratch: int = 10,
    epochs_fe: int = 8,
    epochs_ft: int = 6,
    batch_size: int = config.BATCH_SIZE,
):
    print("=" * 80)
    print("FRESHGUARD AI - COMPLETE TRANSFER LEARNING EXPERIMENT BENCHMARK")
    print("=" * 80)

    total_start = time.time()

    # Experiment A: Scratch CNN
    print("\n>>> STEP 1/4: Training Baseline Scratch CNN...")
    run_scratch(epochs=epochs_scratch, batch_size=batch_size)

    # Experiment B: Feature Extraction
    print("\n>>> STEP 2/4: Training MobileNetV2 Feature Extraction...")
    run_feature_extraction(epochs=epochs_fe, batch_size=batch_size)

    # Experiment C: Fine-Tuning
    print("\n>>> STEP 3/4: Training MobileNetV2 Fine-Tuning...")
    run_finetuning(epochs=epochs_ft, batch_size=batch_size)

    # Step 4: Comprehensive Held-Out Evaluation
    print("\n>>> STEP 4/4: Evaluating All Models on Held-Out Test Set...")
    results = evaluate_all()

    total_duration = time.time() - total_start
    print("=" * 80)
    print(f"BENCHMARK COMPLETED IN {total_duration/60:.2f} MINUTES!")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete FreshGuard AI training and evaluation benchmark")
    parser.add_argument("--epochs-scratch", type=int, default=10, help="Epochs for Scratch CNN")
    parser.add_argument("--epochs-fe", type=int, default=8, help="Epochs for Feature Extraction")
    parser.add_argument("--epochs-ft", type=int, default=6, help="Epochs for Fine-Tuning")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    run_pipeline(
        epochs_scratch=args.epochs_scratch,
        epochs_fe=args.epochs_fe,
        epochs_ft=args.epochs_ft,
        batch_size=args.batch_size,
    )
