"""
FreshGuard AI - Dataset Validation & Integrity Checker
Validates image files, detects corruption, checks duplicates, prevents data leakage,
and computes class distribution statistics.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image
import numpy as np

import config


def get_image_hash(file_path: Path) -> str:
    """Computes MD5 hash of an image file to detect duplicate images."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def verify_image(file_path: Path) -> Tuple[bool, str]:
    """
    Verifies if an image can be opened and decoded properly without corruption.
    Returns (is_valid, error_message).
    """
    if file_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return False, f"Unsupported extension: {file_path.suffix}"
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Verify decoding pixel data
        with Image.open(file_path) as img:
            img.convert("RGB")
        return True, "OK"
    except Exception as e:
        return False, str(e)


def discover_classes(data_dir: Path) -> List[str]:
    """Discovers subdirectories as class names, sorted alphabetically."""
    if not data_dir.exists():
        return []
    classes = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return sorted(classes)


def scan_split_directory(split_dir: Path) -> Tuple[Dict[str, int], List[Path], List[Path], Dict[str, List[Path]]]:
    """
    Scans a dataset directory (train/val/test).
    Returns:
        - class_counts: {class_name: count}
        - valid_files: list of valid Path objects
        - corrupted_files: list of corrupted Path objects
        - duplicates: {hash: [list of Paths with same hash]}
    """
    class_counts: Dict[str, int] = {}
    valid_files: List[Path] = []
    corrupted_files: List[Path] = []
    hash_map: Dict[str, List[Path]] = {}

    if not split_dir.exists():
        return class_counts, valid_files, corrupted_files, {}

    classes = discover_classes(split_dir)
    for cls in classes:
        cls_path = split_dir / cls
        files = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS]
        valid_in_cls = 0

        for f in files:
            is_valid, _ = verify_image(f)
            if is_valid:
                valid_files.append(f)
                valid_in_cls += 1
                img_hash = get_image_hash(f)
                hash_map.setdefault(img_hash, []).append(f)
            else:
                corrupted_files.append(f)

        class_counts[cls] = valid_in_cls

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    return class_counts, valid_files, corrupted_files, duplicates


def check_data_leakage(train_dir: Path, val_dir: Path, test_dir: Path) -> Dict[str, any]:
    """
    Verifies that no images in the test set or validation set appear in the training set
    (checks MD5 hash collisions across splits).
    """
    def build_hash_set(directory: Path) -> Dict[str, Path]:
        hashes = {}
        if not directory.exists():
            return hashes
        for cls_dir in directory.iterdir():
            if cls_dir.is_dir():
                for f in cls_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                        try:
                            h = get_image_hash(f)
                            hashes[h] = f
                        except Exception:
                            continue
        return hashes

    train_hashes = build_hash_set(train_dir)
    val_hashes = build_hash_set(val_dir)
    test_hashes = build_hash_set(test_dir)

    train_val_overlap = set(train_hashes.keys()) & set(val_hashes.keys())
    train_test_overlap = set(train_hashes.keys()) & set(test_hashes.keys())
    val_test_overlap = set(val_hashes.keys()) & set(test_hashes.keys())

    has_leakage = (len(train_val_overlap) > 0) or (len(train_test_overlap) > 0)

    return {
        "has_leakage": has_leakage,
        "train_count": len(train_hashes),
        "val_count": len(val_hashes),
        "test_count": len(test_hashes),
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "leakage_details": [
            f"Train-Val leak: {train_hashes[h].name} == {val_hashes[h].name}" for h in list(train_val_overlap)[:5]
        ] + [
            f"Train-Test leak: {train_hashes[h].name} == {test_hashes[h].name}" for h in list(train_test_overlap)[:5]
        ],
    }


def compute_class_weights(class_counts: Dict[str, int]) -> Dict[int, float]:
    """
    Calculates balanced class weights to handle dataset class imbalance:
    weight = total_samples / (num_classes * count_per_class)
    """
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    if total == 0 or num_classes == 0:
        return {}
    weights = {}
    for idx, (cls, count) in enumerate(sorted(class_counts.items())):
        weights[idx] = total / (num_classes * max(count, 1))
    return weights


def get_dataset_summary() -> Dict:
    """
    Returns a complete summary of the dataset across train, val, and test splits.
    """
    train_counts, train_valid, train_corrupt, train_dups = scan_split_directory(config.TRAIN_DIR)
    val_counts, val_valid, val_corrupt, val_dups = scan_split_directory(config.VAL_DIR)
    test_counts, test_valid, test_corrupt, test_dups = scan_split_directory(config.TEST_DIR)

    leakage = check_data_leakage(config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR)

    all_classes = sorted(list(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())))

    summary = {
        "classes": all_classes if all_classes else config.DEFAULT_CLASSES,
        "train": {
            "counts": train_counts,
            "total": sum(train_counts.values()),
            "corrupted": len(train_corrupt),
            "duplicates": len(train_dups),
        },
        "val": {
            "counts": val_counts,
            "total": sum(val_counts.values()),
            "corrupted": len(val_corrupt),
            "duplicates": len(val_dups),
        },
        "test": {
            "counts": test_counts,
            "total": sum(test_counts.values()),
            "corrupted": len(test_corrupt),
            "duplicates": len(test_dups),
        },
        "leakage": leakage,
        "total_images": sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values()),
        "is_ready_for_training": sum(train_counts.values()) > 0 and sum(val_counts.values()) > 0,
    }
    return summary


if __name__ == "__main__":
    print("Running Dataset Validation...")
    summary = get_dataset_summary()
    print(json.dumps(summary, indent=2))
