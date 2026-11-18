"""
Training routines for conditional and unconditional flow models.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unfold.models import FlowModel
from unfold.plotting import save_figure

logger = logging.getLogger(__name__)


def train_epoch(
    model: FlowModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train flow model for one epoch."""
    model.train()
    total_loss = 0.0
    total_events = 0

    for batch in loader:
        optimizer.zero_grad()
        if model.unconditional:
            x, w = batch[0].to(device), batch[-1].to(device)
            log_p = model.log_prob(x)
        else:
            x, y, w = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            log_p = model.log_prob(x, y)

        loss = -(log_p * w).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss encountered during training step, skipping batch.")
            continue

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = len(x)
        total_loss += loss.item() * batch_size
        total_events += batch_size

    return total_loss / max(total_events, 1)


@torch.no_grad()
def evaluate_epoch(
    model: FlowModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate NLL on validation set."""
    model.eval()
    total_loss = 0.0
    total_events = 0

    for batch in loader:
        if model.unconditional:
            x, w = batch[0].to(device), batch[-1].to(device)
            log_p = model.log_prob(x)
        else:
            x, y, w = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            log_p = model.log_prob(x, y)

        loss = -(log_p * w).mean()
        batch_size = len(x)
        total_loss += loss.item() * batch_size
        total_events += batch_size

    return total_loss / max(total_events, 1)


def train_model(
    model: FlowModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    save_path: str,
    log_csv_path: Optional[str] = None,
    plot_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Full training loop with AdamW, CosineAnnealing, and early stopping.

    Returns:
        dict with best_epoch, best_val_loss, and history dict.
    """
    t_cfg = config["training"]
    device_str = t_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")
    model.to(device)

    epochs = t_cfg.get("epochs", 200)
    lr = float(t_cfg.get("learning_rate", 1e-3))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    patience = t_cfg.get("early_stopping_patience", 15)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {"train_nll": [], "val_nll": [], "lr": []}

    logger.info("Starting training on %s (epochs=%d, lr=%.2e)...", device, epochs, lr)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_nll = train_epoch(model, train_loader, optimizer, device, grad_clip=grad_clip)
        val_nll = evaluate_epoch(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - t0

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %3d/%3d [%.1fs] | Train NLL: %8.4f | Val NLL: %8.4f | LR: %.2e",
            epoch, epochs, elapsed, train_nll, val_nll, current_lr
        )

        if val_nll < best_val_loss:
            best_val_loss = val_nll
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info("Saved new best model checkpoint to %s (Val NLL: %.4f)", save_path, val_nll)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, patience)
                break

    # Save CSV history
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_nll", "val_nll", "lr"])
            for ep, (tr, va, l) in enumerate(zip(history["train_nll"], history["val_nll"], history["lr"]), 1):
                writer.writerow([ep, tr, va, l])

    # Plot loss curves
    if plot_prefix:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(history["train_nll"]) + 1), history["train_nll"], label="Train NLL", color="tab:blue")
        ax.plot(range(1, len(history["val_nll"]) + 1), history["val_nll"], label="Val NLL", color="tab:orange")
        ax.scatter([best_epoch], [best_val_loss], color="red", zorder=5, label=f"Best ({best_val_loss:.3f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative Log-Likelihood")
        ax.set_title("Training and Validation NLL Curves")
        ax.legend(frameon=True)
        save_figure(fig, plot_prefix)

    # Load best checkpoint weights back into model
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
