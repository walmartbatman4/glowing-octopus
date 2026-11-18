"""
PyTorch Datasets and DataLoaders for conditional unfolding.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class UnfoldingDataset(Dataset):
    """PyTorch Dataset for paired truth and reco features."""

    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        if isinstance(x, np.ndarray):
            self.x = torch.from_numpy(x).float()
        else:
            self.x = x.float()

        if y is not None:
            if isinstance(y, np.ndarray):
                self.y = torch.from_numpy(y).float()
            else:
                self.y = y.float()
        else:
            self.y = None

        if weights is not None:
            if isinstance(weights, np.ndarray):
                self.weights = torch.from_numpy(weights).float()
            else:
                self.weights = weights.float()
        else:
            self.weights = torch.ones(len(self.x), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.y is not None:
            return self.x[idx], self.y[idx], self.weights[idx]
        return self.x[idx], self.weights[idx]


def get_dataloaders(
    x_train: torch.Tensor,
    y_train: Optional[torch.Tensor],
    x_val: torch.Tensor,
    y_val: Optional[torch.Tensor],
    batch_size: int = 1024,
    weights_train: Optional[torch.Tensor] = None,
    weights_val: Optional[torch.Tensor] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_ds = UnfoldingDataset(x_train, y_train, weights_train)
    val_ds = UnfoldingDataset(x_val, y_val, weights_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
