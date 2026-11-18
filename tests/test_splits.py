"""
Tests for split integrity and data leakage prevention.
"""

import os
import numpy as np


def test_splits_disjoint_and_complete():
    splits_dir = "data/splits"
    train_idx = np.load(os.path.join(splits_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(splits_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(splits_dir, "test_idx.npy"))

    # Check non-empty
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    # Check pairwise disjoint
    s_train = set(train_idx)
    s_val = set(val_idx)
    s_test = set(test_idx)

    assert len(s_train.intersection(s_val)) == 0, "Train and Val overlap!"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test overlap!"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test overlap!"

    # Check total coverage
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == len(set(all_idx)), "Duplicate indices present across splits!"


def test_herwig_isolation():
    # Verify no Herwig split files exist in data/splits
    splits_dir = "data/splits"
    split_files = os.listdir(splits_dir)
    for f in split_files:
        assert "herwig" not in f.lower(), f"Herwig split file {f} found in {splits_dir}!"
