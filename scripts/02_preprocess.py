"""
Script 02: Preprocess raw arrays into 2D and 6D datasets.
Creates deterministic 70/15/15 event splits and fits transforms on Pythia26 train only.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml

from unfold.features import build_feature_arrays
from unfold.io import load_raw_dataset
from unfold.transforms import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("02_preprocess")


def create_splits(n_events: int, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42):
    """Create deterministic disjoint index splits."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_events)
    rng.shuffle(indices)

    n_train = int(train_frac * n_events)
    n_val = int(val_frac * n_events)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split datasets.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Data configuration file.")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode (20,000 events).")
    parser.add_argument("--num-data", type=int, default=None, help="Number of events to load.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    dr_max = cfg["matching"]["dr_max"]
    splits_dir = cfg["splits_dir"]
    proc_dir = cfg["processed_dir"]
    raw_dir = cfg["data_dir"]
    ckpt_dir = "results/checkpoints"

    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    num_data = 20000 if args.smoke else args.num_data

    logger.info("Loading Pythia26 for preprocessing (num_data=%s)...", num_data)
    pythia_raw = load_raw_dataset("Pythia26", num_data=num_data, cache_dir=raw_dir)

    logger.info("Loading Herwig for preprocessing (num_data=%s)...", num_data)
    herwig_raw = load_raw_dataset("Herwig", num_data=num_data, cache_dir=raw_dir)

    n_pythia = len(pythia_raw["gen_jets"])
    train_idx, val_idx, test_idx = create_splits(
        n_pythia,
        train_frac=cfg["splits"]["train_frac"],
        val_frac=cfg["splits"]["val_frac"],
        seed=seed,
    )

    # Save split index files
    np.save(os.path.join(splits_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(splits_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(splits_dir, "test_idx.npy"), test_idx)
    logger.info("Saved split indices: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Process 2D and 6D datasets
    modes = ["2d", "6d"]
    for mode in modes:
        logger.info("Processing %s dataset...", mode)
        obs_cfg = cfg[f"observables_{mode}"]
        targets = obs_cfg["target"]
        conds = obs_cfg["conditioning"]

        # Build Pythia arrays
        x_pythia, y_pythia, match_pythia, w_pythia = build_feature_arrays(
            pythia_raw, targets, conds, dr_max=dr_max
        )
        pythia_out = os.path.join(proc_dir, f"pythia26_{mode}.npz")
        np.savez_compressed(
            pythia_out,
            x_gen=x_pythia,
            y_sim=y_pythia,
            matched=match_pythia,
            weights=w_pythia,
            target_names=targets,
            conditioning_names=conds,
        )
        logger.info("Saved Pythia26 %s to %s", mode, pythia_out)

        # Build Herwig arrays
        x_herwig, y_herwig, match_herwig, w_herwig = build_feature_arrays(
            herwig_raw, targets, conds, dr_max=dr_max
        )
        herwig_out = os.path.join(proc_dir, f"herwig_{mode}.npz")
        np.savez_compressed(
            herwig_out,
            x_gen=x_herwig,
            y_sim=y_herwig,
            matched=match_herwig,
            weights=w_herwig,
            target_names=targets,
            conditioning_names=conds,
        )
        logger.info("Saved Herwig %s to %s", mode, herwig_out)

        # Fit transforms strictly on Pythia26 train split (matched events only)
        matched_train_mask = match_pythia[train_idx]
        train_matched_idx = train_idx[matched_train_mask]

        logger.info(
            "Fitting %s transforms on Pythia26 train matched (%d events)...",
            mode,
            len(train_matched_idx),
        )
        t_cfg = cfg["transforms_config"]

        pipe_x = Pipeline(targets, t_cfg)
        pipe_x.fit(x_pythia[train_matched_idx])
        pipe_x.save(os.path.join(ckpt_dir, f"transform_x_{mode}.pt"))

        pipe_y = Pipeline(conds, t_cfg)
        pipe_y.fit(y_pythia[train_matched_idx])
        pipe_y.save(os.path.join(ckpt_dir, f"transform_y_{mode}.pt"))
        logger.info("Saved fitted pipelines for %s to %s", mode, ckpt_dir)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
