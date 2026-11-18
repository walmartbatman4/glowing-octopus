"""
Flow models for conditional and unconditional unbinned unfolding.
Uses Zuko Neural Spline Flows (NSF) with optional MLP context embedding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import zuko

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron for context embeddings."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        activation: type = nn.ReLU,
    ):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_features:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowModel(nn.Module):
    """Wrapper around conditional and unconditional normalizing flows."""

    def __init__(
        self,
        dim_x: int,
        dim_y: int = 0,
        unconditional: bool = False,
        flow_type: str = "nsf",
        num_transforms: int = 6,
        hidden_features: Optional[List[int]] = None,
        num_bins: int = 8,
        context_embedding_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.unconditional = unconditional
        self.flow_type = flow_type.lower()
        if hidden_features is None:
            hidden_features = [128, 128, 128]

        # Context embedding network
        if not self.unconditional and dim_y > 0 and context_embedding_cfg and context_embedding_cfg.get("enabled", False):
            emb_hidden = context_embedding_cfg.get("hidden_features", [64, 64])
            emb_out = emb_hidden[-1]
            self.context_net = MLP(dim_y, emb_out, emb_hidden[:-1])
            flow_context_dim = emb_out
        else:
            self.context_net = None
            flow_context_dim = 0 if self.unconditional else dim_y

        # Build base flow
        if self.flow_type == "nsf":
            self.flow = zuko.flows.NSF(
                features=dim_x,
                context=flow_context_dim,
                transforms=num_transforms,
                hidden_features=hidden_features,
                bins=num_bins,
            )
        elif self.flow_type == "affine" or self.flow_type == "maf":
            self.flow = zuko.flows.MAF(
                features=dim_x,
                context=flow_context_dim,
                transforms=num_transforms,
                hidden_features=hidden_features,
            )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

    def _get_context(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.unconditional:
            return None
        if y is None:
            raise ValueError("Context 'y' must be provided for conditional flow.")
        if self.context_net is not None:
            return self.context_net(y)
        return y

    def log_prob(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log p(x|y) or log p(x)."""
        ctx = self._get_context(y)
        if ctx is None:
            dist = self.flow()
        else:
            dist = self.flow(ctx)
        return dist.log_prob(x)

    def sample(self, y: Optional[torch.Tensor] = None, num_samples: Optional[int] = None) -> torch.Tensor:
        """Sample x ~ p(x|y) or x ~ p(x).

        Args:
            y: conditioning tensor of shape (N, Dy) if conditional.
            num_samples: K samples per event. If None, draws 1 sample per event.

        Returns:
            If num_samples is None: tensor of shape (N, Dx).
            If num_samples is K: tensor of shape (K, N, Dx) or (K, Dx) for unconditional.
        """
        ctx = self._get_context(y)
        if ctx is None:
            dist = self.flow()
            shape = (num_samples,) if num_samples is not None else ()
            return dist.sample(shape)
        else:
            dist = self.flow(ctx)
            if num_samples is None or num_samples == 1:
                s = dist.sample()
                return s
            else:
                return dist.sample((num_samples,))

    def transform_to_latent(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Push-forward x through the flow to latent base space z = f(x; y)."""
        ctx = self._get_context(y)
        if ctx is None:
            dist = self.flow()
        else:
            dist = self.flow(ctx)
        return dist.transform(x)


def build_model(config: Dict[str, Any], unconditional: bool = False) -> FlowModel:
    """Build FlowModel from configuration dict."""
    m_cfg = config["model"]
    return FlowModel(
        dim_x=m_cfg["dim_x"],
        dim_y=m_cfg.get("dim_y", 0),
        unconditional=unconditional,
        flow_type=m_cfg.get("type", "nsf"),
        num_transforms=m_cfg.get("num_transforms", 6),
        hidden_features=m_cfg.get("hidden_features", [128, 128, 128]),
        num_bins=m_cfg.get("num_bins", 8),
        context_embedding_cfg=m_cfg.get("context_embedding", None),
    )
