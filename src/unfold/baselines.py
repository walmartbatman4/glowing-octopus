"""
Baseline unfolding methods: Reco-as-answer, Prior-only, and Iterative Bayesian Unfolding (IBU).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyunfold import iterative_unfold

logger = logging.getLogger(__name__)


def reco_as_answer(
    y_sim: np.ndarray,
    target_indices_in_y: List[int],
) -> np.ndarray:
    """Reco-as-answer baseline: use detector-level observables directly as estimate of truth.

    Args:
        y_sim: Sim/reco array of shape (N, Dy).
        target_indices_in_y: Column indices in y corresponding to the targets.

    Returns:
        Array of shape (N, Dx).
    """
    return y_sim[:, target_indices_in_y]


def prior_only(
    x_train_gen: np.ndarray,
    num_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Prior-only baseline: bootstrap-resample training truth, completely ignoring conditioning."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_train_gen), size=num_samples, replace=True)
    return x_train_gen[idx]


class IBU2D:
    """Iterative Bayesian Unfolding (IBU) on a flattened 2D grid using PyUnfold."""

    def __init__(
        self,
        bins_per_dim: int = 20,
        num_iterations: int = 4,
    ):
        self.bins_per_dim = bins_per_dim
        self.num_iterations = num_iterations
        self.edges_x: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.edges_y: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.response_matrix: Optional[np.ndarray] = None
        self.response_err: Optional[np.ndarray] = None
        self.efficiencies: Optional[np.ndarray] = None
        self.efficiencies_err: Optional[np.ndarray] = None
        self.num_cause_bins: int = bins_per_dim * bins_per_dim
        self.num_effect_bins: int = bins_per_dim * bins_per_dim

    def fit_response(
        self,
        x_train_gen: np.ndarray,
        y_train_sim: np.ndarray,
        matched_train: np.ndarray,
    ) -> "IBU2D":
        """Build the 2D flattened response matrix and efficiencies from training data."""
        # Fit bin edges from quantiles of training matched data
        x_m = x_train_gen[matched_train]
        y_m = y_train_sim[matched_train]

        q = np.linspace(0.001, 0.999, self.bins_per_dim + 1)
        edges_x0 = np.quantile(x_m[:, 0], q)
        edges_x1 = np.quantile(x_m[:, 1], q)
        # Ensure strictly monotonic
        edges_x0 = np.unique(edges_x0)
        edges_x1 = np.unique(edges_x1)
        if len(edges_x0) < self.bins_per_dim + 1:
            edges_x0 = np.linspace(x_m[:, 0].min(), x_m[:, 0].max(), self.bins_per_dim + 1)
        if len(edges_x1) < self.bins_per_dim + 1:
            edges_x1 = np.linspace(x_m[:, 1].min(), x_m[:, 1].max(), self.bins_per_dim + 1)

        edges_y0 = np.linspace(y_m[:, 0].min(), y_m[:, 0].max(), self.bins_per_dim + 1)
        edges_y1 = np.linspace(y_m[:, 1].min(), y_m[:, 1].max(), self.bins_per_dim + 1)

        self.edges_x = (edges_x0, edges_x1)
        self.edges_y = (edges_y0, edges_y1)

        # Digitize matched cause (gen) and effect (sim) into 1D bin indices
        b_x0 = np.clip(np.digitize(x_m[:, 0], edges_x0) - 1, 0, self.bins_per_dim - 1)
        b_x1 = np.clip(np.digitize(x_m[:, 1], edges_x1) - 1, 0, self.bins_per_dim - 1)
        cause_bins_m = b_x0 * self.bins_per_dim + b_x1

        b_y0 = np.clip(np.digitize(y_m[:, 0], edges_y0) - 1, 0, self.bins_per_dim - 1)
        b_y1 = np.clip(np.digitize(y_m[:, 1], edges_y1) - 1, 0, self.bins_per_dim - 1)
        effect_bins_m = b_y0 * self.bins_per_dim + b_y1

        # Response counts matrix N(effect j, cause i)
        n_cause = self.bins_per_dim * self.bins_per_dim
        n_effect = self.bins_per_dim * self.bins_per_dim
        hist_2d, _, _ = np.histogram2d(
            effect_bins_m,
            cause_bins_m,
            bins=[np.arange(n_effect + 1), np.arange(n_cause + 1)],
        )

        # Efficiency: matched causes / total generated causes in bin
        b_all_x0 = np.clip(np.digitize(x_train_gen[:, 0], edges_x0) - 1, 0, self.bins_per_dim - 1)
        b_all_x1 = np.clip(np.digitize(x_train_gen[:, 1], edges_x1) - 1, 0, self.bins_per_dim - 1)
        cause_all_bins = b_all_x0 * self.bins_per_dim + b_all_x1
        total_cause_counts = np.bincount(cause_all_bins, minlength=n_cause)

        matched_cause_counts = hist_2d.sum(axis=0)  # sum over effects
        efficiencies = np.divide(
            matched_cause_counts,
            total_cause_counts,
            out=np.zeros(n_cause),
            where=(total_cause_counts > 0),
        )
        efficiencies = np.clip(efficiencies, 1e-4, 1.0)
        efficiencies_err = np.sqrt(efficiencies * (1.0 - efficiencies) / np.maximum(total_cause_counts, 1.0))
        efficiencies_err = np.maximum(efficiencies_err, 1e-4)

        # Normalise response matrix: P(effect j | cause i)
        col_sums = hist_2d.sum(axis=0, keepdims=True)
        response = np.divide(hist_2d, col_sums, out=np.zeros_like(hist_2d), where=(col_sums > 0))
        # Ensure columns without counts have uniform fallback
        empty_cols = (col_sums.squeeze() == 0)
        response[:, empty_cols] = 1.0 / n_effect

        # Poisson error on response
        response_err = np.divide(
            np.sqrt(np.maximum(hist_2d, 1.0)),
            np.maximum(col_sums, 1.0),
        )
        response_err = np.maximum(response_err, 1e-4)

        self.response_matrix = response
        self.response_err = response_err
        self.efficiencies = efficiencies
        self.efficiencies_err = efficiencies_err
        return self

    def unfold(
        self,
        y_test_sim: np.ndarray,
    ) -> Dict[str, Any]:
        """Unfold observed test reco data using PyUnfold (or exact vectorized D'Agostini if >100 bins)."""
        if self.edges_y is None or self.response_matrix is None:
            raise RuntimeError("IBU2D response matrix not fitted!")

        edges_y0, edges_y1 = self.edges_y
        b_y0 = np.clip(np.digitize(y_test_sim[:, 0], edges_y0) - 1, 0, self.bins_per_dim - 1)
        b_y1 = np.clip(np.digitize(y_test_sim[:, 1], edges_y1) - 1, 0, self.bins_per_dim - 1)
        effect_bins = b_y0 * self.bins_per_dim + b_y1

        n_effect = self.bins_per_dim * self.bins_per_dim
        n_cause = n_effect
        data_counts = np.bincount(effect_bins, minlength=n_effect).astype(float)
        data_err = np.sqrt(np.maximum(data_counts, 1.0))

        # PyUnfold's covariance implementation allocates an (E*C, E*C) matrix which requires
        # ~200 GB RAM when bins > 100. Use PyUnfold when bins <= 100, and exact D'Agostini when bins > 100.
        if n_cause <= 100:
            unfolded_result = iterative_unfold(
                data=data_counts,
                data_err=data_err,
                response=self.response_matrix,
                response_err=self.response_err,
                efficiencies=self.efficiencies,
                efficiencies_err=self.efficiencies_err,
                max_iter=self.num_iterations,
                ts="ks",
                ts_stopping=0.001,
            )
            unfolded_counts = unfolded_result["unfolded"]
            unfolded_err = unfolded_result.get("stat_err", np.sqrt(np.maximum(unfolded_counts, 1.0)))
            num_iters = unfolded_result.get("num_iterations", self.num_iterations)
        else:
            # Exact vectorized D'Agostini iterative unfolding
            eff_inv = 1.0 / np.maximum(self.efficiencies, 1e-6)
            # Uniform initial prior
            n_c = np.full(n_cause, np.sum(data_counts) / n_cause)
            Mij = np.zeros_like(self.response_matrix)

            for it in range(self.num_iterations):
                # Denominator f_j = sum_i P(E_j | C_i) * n_{c_i}
                f_norm = np.dot(self.response_matrix, n_c)
                f_inv = 1.0 / np.maximum(f_norm, 1e-12)
                # Unfolding matrix Mij = P(E_j | C_i) * (n_{c_i} / eff_i) / f_j
                n_c_eff = n_c * eff_inv
                Mij = self.response_matrix * n_c_eff.reshape(1, -1) * f_inv.reshape(-1, 1)
                n_c = np.dot(data_counts, Mij)

            unfolded_counts = n_c
            # Statistical error propagation: sigma^2(c_i) = sum_j M_{ji}^2 * sigma^2(e_j)
            unfolded_err = np.sqrt(np.dot(data_err**2, Mij**2))
            num_iters = self.num_iterations

        # Project 1D bin counts to marginals along dim0 and dim1
        counts_2d = unfolded_counts.reshape(self.bins_per_dim, self.bins_per_dim)
        err_2d = unfolded_err.reshape(self.bins_per_dim, self.bins_per_dim)

        marginal_0 = counts_2d.sum(axis=1)
        marginal_0_err = np.sqrt((err_2d**2).sum(axis=1))

        marginal_1 = counts_2d.sum(axis=0)
        marginal_1_err = np.sqrt((err_2d**2).sum(axis=0))

        return {
            "unfolded_flat": unfolded_counts,
            "unfolded_err": unfolded_err,
            "counts_2d": counts_2d,
            "err_2d": err_2d,
            "marginal_0": marginal_0,
            "marginal_0_err": marginal_0_err,
            "marginal_1": marginal_1,
            "marginal_1_err": marginal_1_err,
            "edges_x0": self.edges_x[0],
            "edges_x1": self.edges_x[1],
            "iterations": num_iters,
        }
