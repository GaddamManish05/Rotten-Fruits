# FreshGuard AI - Outputs Directory

This directory stores evaluation metrics, training histories, learning curves, and confusion matrix plots.

## Subdirectories
- `figures/`: High-resolution figures generated during training and evaluation:
  - `curves_scratch.png`: Accuracy & loss curves for Scratch CNN.
  - `curves_feature_extraction.png`: Curves for MobileNetV2 Feature Extraction.
  - `curves_finetuned.png`: Curves for MobileNetV2 Fine-Tuning.
  - `cm_*.png`: Confusion matrices for each evaluated model.
- `metrics/`: JSON files containing machine-readable metrics:
  - `history_scratch.json`: Epoch-by-epoch loss, accuracy, and training duration.
  - `history_feature_extraction.json`: Feature extraction training history.
  - `history_finetuned.json`: Fine-tuning training history.
  - `model_comparison.json`: Comprehensive held-out test evaluation benchmark table.
