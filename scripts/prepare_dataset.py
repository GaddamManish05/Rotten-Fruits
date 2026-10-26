"""
FreshGuard AI - Dataset Preparation Pipeline
Downloads or sources the Kaggle Fresh and Rotten Fruit dataset,
splits into verified train/val/test folders, prevents data leakage,
removes corrupted files, and computes distribution statistics.
"""

import argparse
import hashlib
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset_validation import verify_image, get_image_hash, get_dataset_summary


def download_or_find_dataset(custom_path: str = None) -> Path:
    """Locates the dataset locally or downloads it via kagglehub."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        print(f"Warning: Custom path {custom_path} not found. Checking cache/kagglehub...")

    # Check kagglehub cache default path
    cache_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "sriramr" / "fruits-fresh-and-rotten-for-classification" / "versions" / "1"
    if cache_path.exists():
        # Might have nested dataset directory
        if (cache_path / "dataset").exists():
            return cache_path / "dataset"
        return cache_path

    # Try downloading via kagglehub
    try:
        import kagglehub
        print("Downloading 'sriramr/fruits-fresh-and-rotten-for-classification' via kagglehub...")
        download_dir = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")
        d_path = Path(download_dir)
        if (d_path / "dataset").exists():
            return d_path / "dataset"
        return d_path
    except Exception as e:
        print(f"Automatic download failed: {e}")
        print("\nPlease manually download the Kaggle dataset:")
        print("https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification")
        print("and place the extracted files in 'data/raw/'")
        raise


def prepare_dataset(
    source_dir: Path,
    mode: str = "quick",
    train_val_split: float = 0.8,
):
    """
    Organizes dataset into data/train, data/val, data/test with strict leakage prevention.
    Modes:
      - 'quick': ~200 train, 50 val, 50 test per class (ideal for student laptops & quick verification)
      - 'medium': ~500 train, 100 val, 100 test per class
      - 'full': All available images
    """
    print("=" * 60)
    print("FreshGuard AI - Dataset Preparation Pipeline")
    print(f"Source Directory: {source_dir}")
    print(f"Mode: {mode}")
    print("=" * 60)

    # Locate source train and test directories
    raw_train = None
    raw_test = None

    if (source_dir / "train").exists() and (source_dir / "test").exists():
        raw_train = source_dir / "train"
        raw_test = source_dir / "test"
    elif (source_dir / "dataset" / "train").exists():
        raw_train = source_dir / "dataset" / "train"
        raw_test = source_dir / "dataset" / "test"
    else:
        # Check subdirectories
        subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
        for d in subdirs:
            if (d / "train").exists() and (d / "test").exists():
                raw_train = d / "train"
                raw_test = d / "test"
                break

    if raw_train is None or raw_test is None:
        raise FileNotFoundError(f"Could not locate 'train' and 'test' subdirectories in {source_dir}")

    # Discover classes
    classes = [d.name for d in raw_train.iterdir() if d.is_dir() and not d.name.startswith(".")]
    classes = sorted(classes)
    print(f"Discovered {len(classes)} classes: {classes}")

    # Set sample limits based on mode
    limits = {
        "quick": {"train": 200, "val": 50, "test": 50},
        "medium": {"train": 500, "val": 120, "test": 100},
        "full": {"train": 999999, "val": 999999, "test": 999999},
    }
    lim = limits.get(mode, limits["quick"])

    # Clear destination directories
    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    random.seed(config.RANDOM_SEED)

    counts = {"train": {}, "val": {}, "test": {}}

    for cls in classes:
        print(f"\nProcessing class: {cls}...")
        cls_train_src = raw_train / cls
        cls_test_src = raw_test / cls

        # Collect and verify test candidates
        test_files = [f for f in cls_test_src.iterdir() if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS]
        random.shuffle(test_files)

        test_copied = 0
        for f in test_files:
            if test_copied >= lim["test"]:
                break
            is_valid, _ = verify_image(f)
            if not is_valid:
                continue
            h = get_image_hash(f)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            shutil.copy2(f, config.TEST_DIR / cls / f.name)
            test_copied += 1
        counts["test"][cls] = test_copied

        # Collect and verify train/val candidates
        train_src_files = [f for f in cls_train_src.iterdir() if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS]
        random.shuffle(train_src_files)

        val_copied = 0
        train_copied = 0

        # We first fill val, then train, ensuring strict separation
        for f in train_src_files:
            if val_copied >= lim["val"] and train_copied >= lim["train"]:
                break

            is_valid, _ = verify_image(f)
            if not is_valid:
                continue

            h = get_image_hash(f)
            if h in seen_hashes:
                # Discard duplicate to prevent leakage
                continue
            seen_hashes.add(h)

            if val_copied < lim["val"]:
                shutil.copy2(f, config.VAL_DIR / cls / f.name)
                val_copied += 1
            elif train_copied < lim["train"]:
                shutil.copy2(f, config.TRAIN_DIR / cls / f.name)
                train_copied += 1

        counts["val"][cls] = val_copied
        counts["train"][cls] = train_copied

        print(f"  -> Train: {train_copied} | Val: {val_copied} | Test: {test_copied}")

    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Total Unique Images Copied: {len(seen_hashes)}")
    summary = get_dataset_summary()
    print(f"Leakage Check Passed: {not summary['leakage']['has_leakage']}")
    print(f"Train samples: {summary['train']['total']}")
    print(f"Validation samples: {summary['val']['total']}")
    print(f"Test samples: {summary['test']['total']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FreshGuard AI dataset")
    parser.add_argument("--source", type=str, default=None, help="Custom path to raw dataset")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "medium", "full"],
                        help="Dataset size mode: 'quick' (~1800 imgs), 'medium' (~4300 imgs), 'full' (all)")
    args = parser.parse_args()

    src_dir = download_or_find_dataset(args.source)
    prepare_dataset(src_dir, mode=args.mode)
