# FreshGuard AI - Data Directory

This directory stores raw and partitioned fruit image datasets.

## Directory Structure
- `data/raw/`: Original unpartitioned dataset images or downloaded Kaggle zips.
- `data/train/`: Verified training split (80% of data, used with augmentation).
- `data/val/`: Validation split (20% of data, used for early stopping and LR reduction).
- `data/test/`: Independent held-out test split (used strictly for final evaluation).

## Categories (6 Classes)
1. `freshapples`: Fresh Apples
2. `freshbanana`: Fresh Bananas
3. `freshoranges`: Fresh Oranges
4. `rottenapples`: Rotten Apples
5. `rottenbanana`: Rotten Bananas
6. `rottenoranges`: Rotten Oranges

## Integrity Checks
- All images are verified with Pillow `img.verify()`.
- MD5 image hashing ensures zero cross-split data leakage.
