"""
Sampling and unfolding inference routines.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

from unfold.models import FlowModel
from unfold.transforms import Pipeline

logger = logging.getLogger(__name__)


def unfold(
    model: FlowModel,
    y_reco: Union[np.ndarray, torch.Tensor],
    pipeline_x: Pipeline,
    pipeline_y: Pipeline,
    k: int = 1,
    batch_size: int = 2048,
    device: str = "cuda",
) -> np.ndarray:
    """Perform unbinned unfolding by sampling truth x ~ p(x|y).

    Args:
        model: Trained conditional flow model.
        y_reco: Sim/reco conditioning features in physical units, shape (N, Dy).
        pipeline_x: Fitted pipeline for target features x.
        pipeline_y: Fitted pipeline for conditioning features y.
        k: Number of samples per reco event (default 1).
        batch_size: Inference batch size.
        device: Device to run inference on.

    Returns:
        Unfolded truth features in physical units, shape (N, Dx) if k==1 else (k, N, Dx).
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)

    if isinstance(y_reco, np.ndarray):
        y_tensor = torch.from_numpy(y_reco).float()
    else:
        y_tensor = y_reco.float()

    n_events = len(y_tensor)
    samples_list = []

    with torch.no_grad():
        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            y_batch = y_tensor[start:end]

            # 1. Transform y into normalized latent conditioning space
            y_norm, _ = pipeline_y(y_batch, training=False)
            y_norm = y_norm.to(dev)

            # 2. Draw samples from conditional flow
            if k == 1:
                x_norm_batch = model.sample(y_norm, num_samples=1)
                # Invert back to physical units
                x_phys_batch = pipeline_x.inverse(x_norm_batch.cpu())
                samples_list.append(x_phys_batch.numpy())
            else:
                x_norm_batch = model.sample(y_norm, num_samples=k)  # (K, B, Dx)
                k_samples = []
                for sample_i in range(k):
                    x_phys_i = pipeline_x.inverse(x_norm_batch[sample_i].cpu())
                    k_samples.append(x_phys_i.numpy())
                samples_list.append(np.stack(k_samples, axis=0))  # (K, B, Dx)

    if k == 1:
        return np.concatenate(samples_list, axis=0)
    else:
        return np.concatenate(samples_list, axis=1)


def sample_unconditional(
    model: FlowModel,
    pipeline_x: Pipeline,
    num_samples: int,
    batch_size: int = 2048,
    device: str = "cuda",
) -> np.ndarray:
    """Sample from the unconditional control model p(x)."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)

    samples_list = []
    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            b_size = min(remaining, batch_size)
            x_norm = model.sample(y=None, num_samples=b_size)
            x_phys = pipeline_x.inverse(x_norm.cpu())
            samples_list.append(x_phys.numpy())
            remaining -= b_size

    return np.concatenate(samples_list, axis=0)
