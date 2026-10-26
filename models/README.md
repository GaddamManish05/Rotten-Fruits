# FreshGuard AI - Models Directory

This directory stores trained model checkpoints in standard Keras 3 format (`.keras`).

## Files
- `model_scratch.keras`: Baseline CNN trained from scratch with random weight initialization (Experiment A).
- `model_feature_extraction.keras`: Pretrained MobileNetV2 with frozen backbone and trained classification head (Experiment B).
- `model_finetuned.keras`: MobileNetV2 with deeper layers unfrozen and trained at a low learning rate (\(10^{-5}\)) (Experiment C).
- `model_efficientnet.keras`: Optional comparison model using EfficientNetB0 (Experiment D).

Do not commit large model weights to version control repositories.
