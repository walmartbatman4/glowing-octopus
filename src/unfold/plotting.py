"""
Plotting utilities and figure generators for unfolding.
All figures are saved in both PNG and PDF formats.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Try to use mplhep style if available
try:
    import mplhep as hep
    hep.style.use("CMS")
    HAS_MPLHEP = True
except Exception:
    HAS_MPLHEP = False
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def save_figure(fig: plt.Figure, filepath_no_ext: str, dpi: int = 300) -> None:
    """Save a matplotlib figure to both PNG and PDF."""
    dirname = os.path.dirname(filepath_no_ext)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    png_path = f"{filepath_no_ext}.png"
    pdf_path = f"{filepath_no_ext}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figures to %s and %s", png_path, pdf_path)


def plot_observable_distribution(
    gen_vals: np.ndarray,
    sim_vals: np.ndarray,
    obs_name: str,
    output_prefix: str,
    bins: int = 50,
    range_quantile: Tuple[float, float] = (0.001, 0.999),
) -> None:
    """Plot gen-level vs sim-level 1D marginal with ratio panel."""
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    q_low = min(float(np.quantile(gen_vals, range_quantile[0])), float(np.quantile(sim_vals, range_quantile[0])))
    q_high = max(float(np.quantile(gen_vals, range_quantile[1])), float(np.quantile(sim_vals, range_quantile[1])))
    if q_low == q_high:
        q_low, q_high = float(gen_vals.min()), float(gen_vals.max() + 1e-3)

    bin_edges = np.linspace(q_low, q_high, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    h_gen, _ = np.histogram(gen_vals, bins=bin_edges, density=True)
    h_sim, _ = np.histogram(sim_vals, bins=bin_edges, density=True)

    ax_main.step(bin_centers, h_gen, where="mid", label="Gen (Truth)", color="black", lw=1.8)
    ax_main.step(bin_centers, h_sim, where="mid", label="Sim (Reco)", color="tab:blue", lw=1.5, ls="--")
    ax_main.set_ylabel("Normalized Density")
    ax_main.set_title(f"Gen vs Sim Distribution: {obs_name}")
    ax_main.legend(frameon=True)

    # Ratio plot
    ratio = np.divide(h_sim, h_gen, out=np.ones_like(h_sim), where=(h_gen > 0))
    ax_ratio.step(bin_centers, ratio, where="mid", color="tab:blue", lw=1.5)
    ax_ratio.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Sim / Gen")
    ax_ratio.set_xlabel(obs_name)

    save_figure(fig, output_prefix)


def plot_unfolding_marginal_comparison(
    x_true: np.ndarray,
    x_reco: np.ndarray,
    x_unfolded: np.ndarray,
    obs_name: str,
    output_prefix: str,
    x_ibu: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    bins: int = 35,
) -> None:
    """Plot 1D marginal comparing Truth, Reco, cINN unfolded, and optional IBU."""
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    q_low = float(np.quantile(x_true, 0.002))
    q_high = float(np.quantile(x_true, 0.998))
    bin_edges = np.linspace(q_low, q_high, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    h_true, _ = np.histogram(x_true, bins=bin_edges, density=True)
    h_reco, _ = np.histogram(x_reco, bins=bin_edges, density=True)
    h_unf, _ = np.histogram(x_unfolded, bins=bin_edges, density=True)

    ax_main.step(bin_centers, h_true, where="mid", label="Truth (Gen)", color="black", lw=2.0)
    ax_main.step(bin_centers, h_reco, where="mid", label="Reco (Sim)", color="tab:blue", lw=1.5, ls="--")
    ax_main.step(bin_centers, h_unf, where="mid", label="cINN Unfolded", color="tab:red", lw=2.0)

    # If IBU is provided: (ibu_counts, ibu_err, ibu_bin_edges)
    if x_ibu is not None:
        ibu_counts, ibu_err, ibu_edges = x_ibu
        ibu_total = np.sum(ibu_counts) * bin_width
        if ibu_total > 0:
            ibu_density = ibu_counts / ibu_total
            ibu_density_err = ibu_err / ibu_total
            ax_main.errorbar(
                bin_centers[:len(ibu_density)],
                ibu_density,
                yerr=ibu_density_err,
                fmt="s",
                color="tab:green",
                label="IBU",
                markersize=4,
                capsize=2,
            )

    ax_main.set_ylabel("Normalized Density")
    ax_main.set_title(f"Unfolding Benchmark: {obs_name}")
    ax_main.legend(frameon=True)

    # Ratio panels
    ratio_unf = np.divide(h_unf, h_true, out=np.ones_like(h_unf), where=(h_true > 0))
    ratio_reco = np.divide(h_reco, h_true, out=np.ones_like(h_reco), where=(h_true > 0))

    ax_ratio.step(bin_centers, ratio_reco, where="mid", color="tab:blue", lw=1.5, ls="--", label="Reco/Truth")
    ax_ratio.step(bin_centers, ratio_unf, where="mid", color="tab:red", lw=1.8, label="cINN/Truth")

    if x_ibu is not None:
        ibu_counts, ibu_err, _ = x_ibu
        ibu_total = np.sum(ibu_counts) * bin_width
        if ibu_total > 0:
            ibu_dens = ibu_counts / ibu_total
            h_tr_sub = h_true[:len(ibu_dens)]
            ratio_ibu = np.divide(ibu_dens, h_tr_sub, out=np.ones_like(ibu_dens), where=(h_tr_sub > 0))
            ax_ratio.plot(bin_centers[:len(ibu_dens)], ratio_ibu, "s", color="tab:green", markersize=3)

    ax_ratio.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax_ratio.set_ylim(0.7, 1.3)
    ax_ratio.set_ylabel("Ratio to Truth")
    ax_ratio.set_xlabel(obs_name)

    save_figure(fig, output_prefix)


def plot_2d_contour_overlay(
    x_true: np.ndarray,
    x_unfolded: np.ndarray,
    obs_names: List[str],
    output_prefix: str,
    bins: int = 40,
) -> None:
    """Plot 2D density contour overlay comparing truth vs unfolded."""
    fig, ax = plt.subplots(figsize=(7, 6))

    q_x0 = (float(np.quantile(x_true[:, 0], 0.005)), float(np.quantile(x_true[:, 0], 0.995)))
    q_x1 = (float(np.quantile(x_true[:, 1], 0.005)), float(np.quantile(x_true[:, 1], 0.995)))

    h_true, x_e, y_e = np.histogram2d(x_true[:, 0], x_true[:, 1], bins=bins, range=[q_x0, q_x1], density=True)
    h_unf, _, _ = np.histogram2d(x_unfolded[:, 0], x_unfolded[:, 1], bins=bins, range=[q_x0, q_x1], density=True)

    xc = 0.5 * (x_e[:-1] + x_e[1:])
    yc = 0.5 * (y_e[:-1] + y_e[1:])
    X, Y = np.meshgrid(xc, yc)

    ax.contour(X, Y, h_true.T, levels=5, colors="black", linewidths=1.8)
    ax.contour(X, Y, h_unf.T, levels=5, colors="tab:red", linestyles="dashed", linewidths=1.8)

    # Custom legend
    from matplotlib.lines import Line2D
    lines = [
        Line2D([0], [0], color="black", lw=2, label="Truth (Gen)"),
        Line2D([0], [0], color="tab:red", lw=2, linestyle="--", label="cINN Unfolded"),
    ]
    ax.legend(handles=lines, frameon=True)
    ax.set_xlabel(obs_names[0])
    ax.set_ylabel(obs_names[1])
    ax.set_title(f"2D Contour Overlay: {obs_names[0]} vs {obs_names[1]}")

    save_figure(fig, output_prefix)


def plot_sbc_ranks(
    sbc_results: Dict[str, Any],
    obs_names: List[str],
    output_prefix: str,
) -> None:
    """Plot rank histograms from SBC uniformity test."""
    n_dims = len(obs_names)
    fig, axes = plt.subplots(1, n_dims, figsize=(4.5 * n_dims, 4), sharey=True)
    if n_dims == 1:
        axes = [axes]

    for d, (ax, name) in enumerate(zip(axes, obs_names)):
        res = sbc_results[f"dim_{d}"]
        ranks = res["ranks"]
        chi2 = res["chi2"]
        p_val = res["p_value"]

        ax.hist(ranks, bins=20, density=True, color="tab:purple", alpha=0.7, edgecolor="black")
        expected_density = 1.0 / (max(ranks) + 1 if ranks else 100)
        ax.axhline(expected_density, color="red", linestyle="--", lw=1.5, label="Uniform")
        ax.set_title(f"{name}\n$\\chi^2/\\mathrm{{dof}}={chi2/19:.2f}, p={p_val:.3f}$", fontsize=11)
        ax.set_xlabel("Rank")
        if d == 0:
            ax.set_ylabel("Density")
        ax.legend(frameon=True, fontsize=9)

    save_figure(fig, output_prefix)


def plot_corner_pairwise(
    x_true: np.ndarray,
    x_unfolded: np.ndarray,
    obs_names: List[str],
    output_prefix: str,
    max_points: int = 10000,
) -> None:
    """Plot all pairwise 2D distributions (corner plot) for 6D observables."""
    n_dims = len(obs_names)
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(15, 15))

    sub_idx = np.random.choice(len(x_true), size=min(max_points, len(x_true)), replace=False)
    t_sub = x_true[sub_idx]
    u_sub = x_unfolded[sub_idx]

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
            elif i == j:
                # 1D histogram
                q_low = min(float(np.quantile(t_sub[:, i], 0.005)), float(np.quantile(u_sub[:, i], 0.005)))
                q_high = max(float(np.quantile(t_sub[:, i], 0.995)), float(np.quantile(u_sub[:, i], 0.995)))
                ax.hist(t_sub[:, i], bins=30, range=(q_low, q_high), density=True, histtype="step", color="black", lw=1.5)
                ax.hist(u_sub[:, i], bins=30, range=(q_low, q_high), density=True, histtype="step", color="tab:red", lw=1.5, ls="--")
                if i == n_dims - 1:
                    ax.set_xlabel(obs_names[i])
            else:
                # 2D contour or scatter
                q_x = (float(np.quantile(t_sub[:, j], 0.01)), float(np.quantile(t_sub[:, j], 0.99)))
                q_y = (float(np.quantile(t_sub[:, i], 0.01)), float(np.quantile(t_sub[:, i], 0.99)))
                h_t, xe, ye = np.histogram2d(t_sub[:, j], t_sub[:, i], bins=20, range=[q_x, q_y])
                h_u, _, _ = np.histogram2d(u_sub[:, j], u_sub[:, i], bins=20, range=[q_x, q_y])
                xc = 0.5 * (xe[:-1] + xe[1:])
                yc = 0.5 * (ye[:-1] + ye[1:])
                X, Y = np.meshgrid(xc, yc)
                ax.contour(X, Y, h_t.T, levels=3, colors="black", linewidths=1.0)
                ax.contour(X, Y, h_u.T, levels=3, colors="tab:red", linestyles="dashed", linewidths=1.0)

                if i == n_dims - 1:
                    ax.set_xlabel(obs_names[j])
                if j == 0:
                    ax.set_ylabel(obs_names[i])

    save_figure(fig, output_prefix)
