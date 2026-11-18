"""
Script 03: Train conditional flow (cINN) and unconditional control flow across multiple seeds.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
import yaml

from unfold.datasets import get_dataloaders
from unfold.models import build_model
from unfold.train import train_model
from unfold.transforms import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("03_train")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train cINN and Control flow models.")
    parser.add_argument("--config", type=str, required=True, help="Path to config (e.g. configs/cinn_2d.yaml).")
    parser.add_argument("--smoke", action="store_true", help="Run 2 epochs on 1 seed for fast testing.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mode = cfg["data"]["mode"]
    splits_dir = cfg["data"]["splits_dir"]
    data_path = cfg["data"]["processed_pythia"]
    ckpt_dir = cfg["data"]["checkpoint_dir"]
    fig_dir = "results/figures"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Adjust config if smoke mode
    if args.smoke:
        cfg["training"]["epochs"] = 2
        cfg["training"]["seeds"] = [0]
        cfg["training"]["batch_size"] = min(cfg["training"]["batch_size"], 512)

    seeds = cfg["training"]["seeds"]
    batch_size = cfg["training"]["batch_size"]

    # Load raw preprocessed arrays
    logger.info("Loading preprocessed dataset from %s...", data_path)
    data = np.load(data_path)
    x_gen = data["x_gen"]
    y_sim = data["y_sim"]
    matched = data["matched"]
    weights = data["weights"]
    target_names = list(data["target_names"])
    conditioning_names = list(data["conditioning_names"])

    # Load split indices
    train_idx = np.load(os.path.join(splits_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(splits_dir, "val_idx.npy"))

    # Matched-only for training
    train_matched = train_idx[matched[train_idx]]
    val_matched = val_idx[matched[val_idx]]
    logger.info("Matched events: train=%d, val=%d", len(train_matched), len(val_matched))

    # Load fitted pipelines
    pipe_x_path = os.path.join(ckpt_dir, f"transform_x_{mode}.pt")
    pipe_y_path = os.path.join(ckpt_dir, f"transform_y_{mode}.pt")

    pipe_x = Pipeline.load_from_checkpoint(pipe_x_path)
    pipe_y = Pipeline.load_from_checkpoint(pipe_y_path)

    # Transform data into normalized latent space
    logger.info("Transforming train and val data through pipelines...")
    with torch.no_grad():
        x_tr_norm, _ = pipe_x(x_gen[train_matched], training=True)
        y_tr_norm, _ = pipe_y(y_sim[train_matched], training=True)
        w_tr = torch.from_numpy(weights[train_matched]).float()

        x_va_norm, _ = pipe_x(x_gen[val_matched], training=False)
        y_va_norm, _ = pipe_y(y_sim[val_matched], training=False)
        w_va = torch.from_numpy(weights[val_matched]).float()

    # Train each seed for both cINN and unconditional Control model
    for seed in seeds:
        set_seed(seed)
        logger.info("==========================================")
        logger.info("Starting training for SEED %d (Mode: %s)", seed, mode)
        logger.info("==========================================")

        train_loader, val_loader = get_dataloaders(
            x_train=x_tr_norm,
            y_train=y_tr_norm,
            x_val=x_va_norm,
            y_val=y_va_norm,
            batch_size=batch_size,
            weights_train=w_tr,
            weights_val=w_va,
        )

        # 1. Conditional Flow (cINN)
        logger.info("--- Training Conditional Flow (cINN) ---")
        cinn_model = build_model(cfg, unconditional=False)
        cinn_ckpt = os.path.join(ckpt_dir, f"cinn_{mode}_seed{seed}.pt")
        cinn_csv = os.path.join(ckpt_dir, f"history_cinn_{mode}_seed{seed}.csv")
        cinn_plot = os.path.join(fig_dir, f"loss_cinn_{mode}_seed{seed}")

        train_model(
            model=cinn_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            save_path=cinn_ckpt,
            log_csv_path=cinn_csv,
            plot_prefix=cinn_plot,
        )

        # 2. Unconditional Control Flow p(x)
        logger.info("--- Training Unconditional Control Flow p(x) ---")
        train_loader_ctrl, val_loader_ctrl = get_dataloaders(
            x_train=x_tr_norm,
            y_train=None,
            x_val=x_va_norm,
            y_val=None,
            batch_size=batch_size,
            weights_train=w_tr,
            weights_val=w_va,
        )

        control_model = build_model(cfg, unconditional=True)
        control_ckpt = os.path.join(ckpt_dir, f"control_{mode}_seed{seed}.pt")
        control_csv = os.path.join(ckpt_dir, f"history_control_{mode}_seed{seed}.csv")
        control_plot = os.path.join(fig_dir, f"loss_control_{mode}_seed{seed}")

        train_model(
            model=control_model,
            train_loader=train_loader_ctrl,
            val_loader=val_loader_ctrl,
            config=cfg,
            save_path=control_ckpt,
            log_csv_path=control_csv,
            plot_prefix=control_plot,
        )

    logger.info("All training runs completed successfully for %s!", mode)


if __name__ == "__main__":
    main()
