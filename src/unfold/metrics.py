"""
Evaluation metrics for detector unfolding:
KS, Wasserstein-1, binned Chi2/ndf, Pair-C2ST, Marginal-C2ST, SBC rank test, Slice test, and Latent diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

from unfold.models import FlowModel

logger = logging.getLogger(__name__)


def ks_statistic(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Compute 2-sample Kolmogorov-Smirnov test statistic."""
    res = stats.ks_2samp(x_true, x_pred)
    return float(res.statistic)


def wasserstein_distance_1d(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Compute 1D Wasserstein-1 (Earth Mover's) distance."""
    return float(stats.wasserstein_distance(x_true, x_pred))


def binned_chi2_ndf(
    counts_pred: np.ndarray,
    counts_true: np.ndarray,
    err_pred: Optional[np.ndarray] = None,
    err_true: Optional[np.ndarray] = None,
) -> Tuple[float, float, int]:
    """Compute binned Chi2 / ndf and p-value.

    Returns:
        chi2_ndf: Chi2 divided by degrees of freedom.
        p_value: chi2 survival function p-value.
        ndf: degrees of freedom (num_bins - 1).
    """
    if err_true is None:
        err_true_sq = np.maximum(counts_true, 1.0)
    else:
        err_true_sq = np.maximum(err_true**2, 1e-4)

    if err_pred is None:
        err_pred_sq = np.maximum(counts_pred, 1.0)
    else:
        err_pred_sq = np.maximum(err_pred**2, 1e-4)

    sigma_sq = err_true_sq + err_pred_sq
    diff_sq = (counts_pred - counts_true) ** 2
    chi2 = float(np.sum(diff_sq / sigma_sq))
    ndf = max(len(counts_true) - 1, 1)
    chi2_ndf = chi2 / ndf
    p_val = float(stats.chi2.sf(chi2, ndf))
    return chi2_ndf, p_val, ndf


def compute_c2st_auc(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    seed: int = 42,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
) -> float:
    """Train LightGBM classifier to separate P from Q and return test ROC AUC."""
    n_p = len(samples_p)
    n_q = len(samples_q)
    x = np.concatenate([samples_p, samples_q], axis=0)
    y = np.concatenate([np.ones(n_p, dtype=int), np.zeros(n_q, dtype=int)], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=seed, stratify=y
    )

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def compute_null_auc(
    samples: np.ndarray,
    null_repeats: int = 20,
    seed: int = 42,
    n_estimators: int = 100,
) -> Tuple[float, float]:
    """Compute empirical null AUC mean and standard deviation by splitting identical distribution in halves."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    null_aucs = []

    for r in range(null_repeats):
        perm = rng.permutation(n)
        half = n // 2
        p0 = samples[perm[:half]]
        p1 = samples[perm[half : 2 * half]]
        auc = compute_c2st_auc(p1, p0, seed=seed + r, n_estimators=n_estimators)
        null_aucs.append(auc)

    return float(np.mean(null_aucs)), float(np.std(null_aucs))


def pair_c2st(
    x_true: np.ndarray,
    y_cond: np.ndarray,
    x_unfolded: np.ndarray,
    null_repeats: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Pair-C2ST evaluating joint distribution P(x, y).

    Distinguishes [x_true, y] from [x_unfolded, y].
    """
    pairs_true = np.concatenate([x_true, y_cond], axis=-1)
    pairs_unfolded = np.concatenate([x_unfolded, y_cond], axis=-1)

    auc = compute_c2st_auc(pairs_true, pairs_unfolded, seed=seed)
    null_mean, null_std = compute_null_auc(pairs_true, null_repeats=null_repeats, seed=seed)

    threshold = null_mean + 2.0 * null_std
    passed = auc <= threshold

    return {
        "auc": auc,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_threshold": threshold,
        "passed": bool(passed),
    }


def marginal_c2st(
    x_true: np.ndarray,
    x_unfolded: np.ndarray,
    null_repeats: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Marginal C2ST evaluating marginal distribution P(x) alone."""
    auc = compute_c2st_auc(x_true, x_unfolded, seed=seed)
    null_mean, null_std = compute_null_auc(x_true, null_repeats=null_repeats, seed=seed)
    threshold = null_mean + 2.0 * null_std
    passed = auc <= threshold

    return {
        "auc": auc,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_threshold": threshold,
        "passed": bool(passed),
    }


def sbc_rank_test(
    model: FlowModel,
    x_true_norm: np.ndarray,
    y_cond_norm: np.ndarray,
    k: int = 100,
    num_events: int = 1000,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulation-Based Calibration (SBC) rank test.

    Draws K samples per event and computes rank of true value per dimension.
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)

    rng = np.random.default_rng(seed)
    m = min(num_events, len(x_true_norm))
    sel_idx = rng.choice(len(x_true_norm), size=m, replace=False)

    x_sub = torch.from_numpy(x_true_norm[sel_idx]).float()
    y_sub = torch.from_numpy(y_cond_norm[sel_idx]).float().to(dev)

    dim_x = x_sub.shape[-1]
    with torch.no_grad():
        if model.unconditional:
            samples = model.sample(y=None, num_samples=k)  # (K, Dx)
            # Expand to (K, M, Dx)
            samples = samples.unsqueeze(1).expand(k, m, dim_x).cpu()
        else:
            samples = model.sample(y=y_sub, num_samples=k).cpu()  # (K, M, Dx)

    # ranks per dimension: count of samples < true_val
    # samples shape (K, M, Dx), x_sub shape (M, Dx)
    ranks = np.zeros((m, dim_x), dtype=int)
    for d in range(dim_x):
        s_d = samples[:, :, d].numpy()  # (K, M)
        x_d = x_sub[:, d].numpy()       # (M,)
        ranks[:, d] = (s_d < x_d).sum(axis=0)

    # Compute chi2 uniformity test on ranks (using 20 bins)
    num_bins = 20
    bin_edges = np.linspace(0, k + 1, num_bins + 1)
    expected_counts = m / num_bins

    results = {}
    for d in range(dim_x):
        counts, _ = np.histogram(ranks[:, d], bins=bin_edges)
        chi2 = float(np.sum((counts - expected_counts) ** 2 / expected_counts))
        p_val = float(stats.chi2.sf(chi2, df=num_bins - 1))
        results[f"dim_{d}"] = {
            "chi2": chi2,
            "p_value": p_val,
            "ranks": ranks[:, d].tolist(),
            "counts": counts.tolist(),
        }

    return results


def slice_test(
    x_true: np.ndarray,
    y_reco: np.ndarray,
    x_unfolded: np.ndarray,
    slice_feature_idx: int = 0,
    num_quantiles: int = 5,
) -> List[Dict[str, Any]]:
    """Slice test: evaluates unfolded vs truth inside quantile slices of conditioning variable."""
    q_edges = np.quantile(y_reco[:, slice_feature_idx], np.linspace(0.0, 1.0, num_quantiles + 1))
    results = []

    dim_x = x_true.shape[-1]
    for b in range(num_quantiles):
        mask = (y_reco[:, slice_feature_idx] >= q_edges[b]) & (
            y_reco[:, slice_feature_idx] <= q_edges[b + 1] if b == num_quantiles - 1 else y_reco[:, slice_feature_idx] < q_edges[b + 1]
        )
        if mask.sum() < 10:
            continue

        slice_true = x_true[mask]
        slice_unf = x_unfolded[mask]

        slice_metrics = {
            "bin": b,
            "q_low": float(q_edges[b]),
            "q_high": float(q_edges[b + 1]),
            "count": int(mask.sum()),
            "ks": [ks_statistic(slice_true[:, d], slice_unf[:, d]) for d in range(dim_x)],
            "w1": [wasserstein_distance_1d(slice_true[:, d], slice_unf[:, d]) for d in range(dim_x)],
        }
        results.append(slice_metrics)

    return results


def latent_check(
    model: FlowModel,
    x_norm: np.ndarray,
    y_norm: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Latent space normality diagnostic: mean in [-0.05, 0.05], std in [0.95, 1.05], |corr| < 0.05."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)

    x_tensor = torch.from_numpy(x_norm).float().to(dev)
    y_tensor = torch.from_numpy(y_norm).float().to(dev) if y_norm is not None else None

    with torch.no_grad():
        z = model.transform_to_latent(x_tensor, y_tensor).cpu().numpy()

    means = np.mean(z, axis=0)
    stds = np.std(z, axis=0)
    corr_mat = np.corrcoef(z, rowvar=False)
    if corr_mat.ndim == 0:
        corr_mat = np.array([[1.0]])

    # Off-diagonal correlations
    dim = z.shape[-1]
    off_diag = []
    for i in range(dim):
        for j in range(i + 1, dim):
            off_diag.append(float(corr_mat[i, j]))

    mean_pass = bool(np.all(np.abs(means) <= 0.05))
    std_pass = bool(np.all((stds >= 0.95) & (stds <= 1.05)))
    corr_pass = bool(np.all(np.abs(off_diag) <= 0.05)) if off_diag else True

    return {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "correlation_matrix": corr_mat.tolist(),
        "max_abs_offdiag_corr": float(np.max(np.abs(off_diag))) if off_diag else 0.0,
        "mean_pass": mean_pass,
        "std_pass": std_pass,
        "corr_pass": corr_pass,
        "overall_pass": bool(mean_pass and std_pass and corr_pass),
    }


def bootstrap_w1_and_hist(
    x_true: np.ndarray,
    x_unfolded: np.ndarray,
    num_bootstrap: int = 50,
    bins: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap test events to compute statistical uncertainty bands for W1 and binned ratios."""
    rng = np.random.default_rng(seed)
    n = len(x_true)
    dim_x = x_true.shape[-1]

    w1_boots = [[] for _ in range(dim_x)]
    ratio_boots = [[] for _ in range(dim_x)]
    bin_centers_list = []

    # Bin edges per dimension
    bin_edges = []
    for d in range(dim_x):
        q_low = min(float(np.quantile(x_true[:, d], 0.001)), float(np.quantile(x_unfolded[:, d], 0.001)))
        q_high = max(float(np.quantile(x_true[:, d], 0.999)), float(np.quantile(x_unfolded[:, d], 0.999)))
        edges = np.linspace(q_low, q_high, bins + 1)
        bin_edges.append(edges)
        bin_centers_list.append((0.5 * (edges[:-1] + edges[1:])).tolist())

    for _ in range(num_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t_sample = x_true[idx]
        u_sample = x_unfolded[idx]

        for d in range(dim_x):
            w1_val = stats.wasserstein_distance(t_sample[:, d], u_sample[:, d])
            w1_boots[d].append(w1_val)

            h_true, _ = np.histogram(t_sample[:, d], bins=bin_edges[d], density=True)
            h_unf, _ = np.histogram(u_sample[:, d], bins=bin_edges[d], density=True)
            ratio = np.divide(h_unf, h_true, out=np.ones_like(h_unf), where=(h_true > 0))
            ratio_boots[d].append(ratio)

    results = {}
    for d in range(dim_x):
        w1_arr = np.array(w1_boots[d])
        ratio_arr = np.array(ratio_boots[d])  # (B, num_bins)

        results[f"dim_{d}"] = {
            "w1_mean": float(np.mean(w1_arr)),
            "w1_std": float(np.std(w1_arr)),
            "bin_centers": bin_centers_list[d],
            "ratio_mean": np.mean(ratio_arr, axis=0).tolist(),
            "ratio_std": np.std(ratio_arr, axis=0).tolist(),
        }

    return results
