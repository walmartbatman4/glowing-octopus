"""
Invertible preprocessing transforms for unfolding.
Every transform implements forward, inverse, and log_det_jacobian.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseTransform(nn.Module):
    """Abstract base class for invertible 1D transforms."""

    def fit(self, x: torch.Tensor) -> "BaseTransform":
        return self

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            y: transformed tensor of same shape as x.
            log_det: log-determinant of Jacobian, same shape as x.
        """
        raise NotImplementedError

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass recovering x from y."""
        raise NotImplementedError


class IdentityTransform(BaseTransform):
    """Identity transform (no-op)."""

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros_like(x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y


class LogTransform(BaseTransform):
    """Logarithmic transform: y = log(x + eps)."""

    def __init__(self, eps: float = 1e-3, fit_eps: bool = False):
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.float32))
        self.fit_eps = fit_eps

    def fit(self, x: torch.Tensor) -> "LogTransform":
        if self.fit_eps:
            pos_x = x[x > 0.0]
            if len(pos_x) > 0:
                fitted_eps = float(pos_x.min()) * 1e-3
                self.eps.copy_(torch.tensor(max(fitted_eps, 1e-6), dtype=torch.float32))
        return self

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clamped_x = torch.clamp(x, min=0.0)
        shifted = clamped_x + self.eps
        y = torch.log(shifted)
        ldj = -torch.log(shifted)
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = torch.exp(y) - self.eps
        return torch.clamp(x, min=0.0)


class Logit01Transform(BaseTransform):
    """Logit transform for variables bounded in [0, scale].

    Maps x in [0, scale] to (delta, 1 - delta), then computes logit.
    """

    def __init__(self, scale: float = 1.0, delta: float = 1e-4):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
        self.register_buffer("delta", torch.tensor(float(delta), dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clamped_x = torch.clamp(x, min=0.0, max=float(self.scale))
        u = clamped_x / self.scale
        v = self.delta + (1.0 - 2.0 * self.delta) * u
        # logit(v) = log(v) - log(1 - v)
        y = torch.log(v) - torch.log1p(-v)
        # dy/dx = (1 - 2*delta) / (scale * v * (1 - v))
        ldj = (
            torch.log(1.0 - 2.0 * self.delta)
            - torch.log(self.scale)
            - torch.log(v)
            - torch.log1p(-v)
        )
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        v = torch.sigmoid(y)
        u = (v - self.delta) / (1.0 - 2.0 * self.delta)
        x = u * self.scale
        return torch.clamp(x, min=0.0, max=float(self.scale))


class DequantizeTransform(BaseTransform):
    """Dequantization transform for discrete count variables (e.g. multiplicity).

    Adds uniform noise U(0, 1) during training, recovers integer via floor.
    """

    def __init__(self, min_val: float = 1.0):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float(min_val), dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if training:
            noise = torch.rand_like(x)
            y = x + noise
        else:
            y = x + 0.5
        ldj = torch.zeros_like(x)
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = torch.floor(y)
        return torch.clamp(x, min=float(self.min_val))


class ZeroInflatedLogitTransform(BaseTransform):
    """Zero-inflated transform with logit mapping.

    For variables with an exact point mass at 0 (e.g. z_g in [0, 0.5]):
    - During training, exact zeros receive small uniform noise U(0, noise_scale).
    - Passed through Logit01Transform.
    - At inference, values mapping back below noise_scale are snapped to 0.0.
    """

    def __init__(
        self,
        scale: float = 0.5,
        delta: float = 1e-4,
        noise_scale: float = 1e-4,
    ):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
        self.register_buffer("delta", torch.tensor(float(delta), dtype=torch.float32))
        self.register_buffer("noise_scale", torch.tensor(float(noise_scale), dtype=torch.float32))
        self.logit_transform = Logit01Transform(scale=scale, delta=delta)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clamped_x = torch.clamp(x, min=0.0, max=float(self.scale))
        if training:
            is_zero = clamped_x <= 0.0
            noise = torch.rand_like(clamped_x) * self.noise_scale
            x_smoothed = torch.where(is_zero, noise, clamped_x)
        else:
            x_smoothed = clamped_x

        y, ldj = self.logit_transform(x_smoothed, training=training)
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = self.logit_transform.inverse(y)
        is_zero = x < (self.noise_scale * 1.5)
        x_snapped = torch.where(is_zero, torch.zeros_like(x), x)
        return torch.clamp(x_snapped, min=0.0, max=float(self.scale))


class StandardizeTransform(BaseTransform):
    """Standardize: y = (x - mean) / std."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(dim, dtype=torch.float32))
        self.fitted = False

    def fit(self, x: torch.Tensor) -> "StandardizeTransform":
        # x shape: (N, D) or (N,)
        if x.ndim == 1:
            mean = x.mean()
            std = x.std()
        else:
            mean = x.mean(dim=0)
            std = x.std(dim=0)
        std = torch.clamp(std, min=1e-6)
        self.mean.copy_(mean)
        self.std.copy_(std)
        self.fitted = True
        return self

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = (x - self.mean) / self.std
        ldj = -torch.log(self.std).expand_as(x)
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


class Pipeline(nn.Module):
    """Full multidimensional preprocessing pipeline for feature vectors."""

    def __init__(
        self,
        feature_names: List[str],
        config: Dict[str, Any],
    ):
        super().__init__()
        self.feature_names = feature_names
        self.config = config
        self.dim = len(feature_names)
        self.transforms = nn.ModuleList()

        for name in feature_names:
            var_cfg = config.get(name, {"type": "identity"})
            t_type = var_cfg.get("type", "identity")

            if t_type == "log":
                t = LogTransform(eps=var_cfg.get("eps", 1e-3))
            elif t_type == "logit01":
                t = Logit01Transform(
                    scale=var_cfg.get("scale", 1.0),
                    delta=var_cfg.get("delta", 1e-4),
                )
            elif t_type == "dequantize":
                t = DequantizeTransform(min_val=var_cfg.get("min_val", 1.0))
            elif t_type == "zero_inflated_logit":
                t = ZeroInflatedLogitTransform(
                    scale=var_cfg.get("scale", 0.5),
                    delta=var_cfg.get("delta", 1e-4),
                    noise_scale=var_cfg.get("noise_scale", 1e-4),
                )
            elif t_type == "identity":
                t = IdentityTransform()
            else:
                raise ValueError(f"Unknown transform type '{t_type}' for {name}")

            self.transforms.append(t)

        self.standardize = StandardizeTransform(dim=self.dim)

    def fit(self, x: Union[np.ndarray, torch.Tensor]) -> "Pipeline":
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Step 1: fit individual 1D transforms if needed
        for i, t in enumerate(self.transforms):
            t.fit(x[:, i])

        # Step 2: forward through 1D transforms (in eval mode, without stochasticity)
        u_cols = []
        for i, t in enumerate(self.transforms):
            u_i, _ = t(x[:, i], training=False)
            u_cols.append(u_i)
        u = torch.stack(u_cols, dim=-1)

        # Step 3: fit standardization on the transformed features
        self.standardize.fit(u)
        return self

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through pipeline.

        Returns:
            z: transformed features of shape (N, D).
            total_ldj: sum of log determinants of Jacobian of shape (N,).
        """
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x_t = torch.from_numpy(x).float()
        else:
            x_t = x.float()

        u_cols = []
        ldj_cols = []
        for i, t in enumerate(self.transforms):
            u_i, ldj_i = t(x_t[:, i], training=training)
            u_cols.append(u_i)
            ldj_cols.append(ldj_i)

        u = torch.stack(u_cols, dim=-1)
        ldj_1d = torch.stack(ldj_cols, dim=-1).sum(dim=-1)

        z, ldj_std = self.standardize(u, training=training)
        total_ldj = ldj_1d + ldj_std.sum(dim=-1)

        return z, total_ldj

    def inverse(
        self,
        z: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Invert transformed features back to physical units."""
        if isinstance(z, np.ndarray):
            z_t = torch.from_numpy(z).float()
        else:
            z_t = z.float()

        u = self.standardize.inverse(z_t)
        x_cols = []
        for i, t in enumerate(self.transforms):
            x_i = t.inverse(u[:, i])
            x_cols.append(x_i)
        return torch.stack(x_cols, dim=-1)

    def save(self, path: str) -> None:
        torch.save(
            {
                "feature_names": self.feature_names,
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_from_checkpoint(cls, path: str) -> "Pipeline":
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "state_dict" in data:
            pipe = cls(data["feature_names"], data["config"])
            pipe.load_state_dict(data["state_dict"])
            return pipe
        else:
            raise ValueError(f"Invalid checkpoint format at {path}")

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "state_dict" in data:
            self.load_state_dict(data["state_dict"])
        else:
            self.load_state_dict(data)
