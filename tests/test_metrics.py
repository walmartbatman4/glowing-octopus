"""
Unit tests for evaluation metrics and baselines.
"""

import numpy as np
import pytest
import torch
from unfold.baselines import prior_only, reco_as_answer
from unfold.metrics import (
    binned_chi2_ndf,
    ks_statistic,
    latent_check,
    marginal_c2st,
    pair_c2st,
    slice_test,
    wasserstein_distance_1d,
)
from unfold.models import FlowModel


def test_1d_metrics():
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 1000)
    x2 = rng.normal(0, 1, 1000)
    x3 = rng.normal(2, 1, 1000)

    # Identical distributions should have small KS and W1
    ks_same = ks_statistic(x1, x2)
    w1_same = wasserstein_distance_1d(x1, x2)
    assert ks_same < 0.1
    assert w1_same < 0.2

    # Shifted distributions should have large KS and W1
    ks_diff = ks_statistic(x1, x3)
    w1_diff = wasserstein_distance_1d(x1, x3)
    assert ks_diff > 0.5
    assert w1_diff > 1.5


def test_binned_chi2():
    counts_true = np.array([100, 200, 150, 50], dtype=float)
    counts_pred_exact = np.copy(counts_true)
    chi2_ndf, p_val, ndf = binned_chi2_ndf(counts_pred_exact, counts_true)
    assert chi2_ndf == 0.0
    assert p_val == 1.0


def test_c2st_null_calibration():
    rng = np.random.default_rng(42)
    x_true = rng.normal(0, 1, (500, 2))
    y_cond = rng.normal(0, 1, (500, 4))
    x_unfolded = x_true + rng.normal(0, 0.01, (500, 2))  # very close

    res = pair_c2st(x_true, y_cond, x_unfolded, null_repeats=5, seed=42)
    assert "auc" in res
    assert "null_mean" in res
    assert "null_threshold" in res
    # With nearly identical pairs, AUC should be close to null (~0.5)
    assert res["auc"] < 0.65


def test_slice_test():
    rng = np.random.default_rng(42)
    x_true = rng.normal(0, 1, (500, 2))
    y_reco = rng.uniform(0, 10, (500, 2))
    x_unfolded = x_true + rng.normal(0, 0.1, (500, 2))

    slices = slice_test(x_true, y_reco, x_unfolded, slice_feature_idx=0, num_quantiles=3)
    assert len(slices) == 3
    for s in slices:
        assert "ks" in s
        assert "w1" in s


def test_latent_check():
    # A standard Gaussian model should pass latent check
    model = FlowModel(dim_x=2, dim_y=0, unconditional=True, num_transforms=1, hidden_features=[32, 32])
    # Pass standard normal directly
    rng = np.random.default_rng(42)
    z_norm = rng.normal(0, 1, (5000, 2))
    # Latent check expects model.transform_to_latent
    # Test on random mock flow
    res = latent_check(model, z_norm, device="cpu")
    assert "means" in res
    assert "stds" in res
    assert "correlation_matrix" in res
