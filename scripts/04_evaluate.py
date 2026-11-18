"""
Script 04: Evaluate unfolding benchmarks on Pythia26 test split.
Computes KS, W1, Pair-C2ST, Marginal-C2ST, IBU (2D), Control flow, Latent checks, SBC ranks, and Slice tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from unfold.baselines import IBU2D, prior_only, reco_as_answer
from unfold.metrics import (
    binned_chi2_ndf,
    bootstrap_w1_and_hist,
    ks_statistic,
    latent_check,
    marginal_c2st,
    pair_c2st,
    sbc_rank_test,
    slice_test,
    wasserstein_distance_1d,
)
from unfold.models import build_model
from unfold.plotting import (
    plot_2d_contour_overlay,
    plot_corner_pairwise,
    plot_sbc_ranks,
    plot_unfolding_marginal_comparison,
)
from unfold.sample import sample_unconditional, unfold
from unfold.transforms import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("04_evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unfolding models on Pythia test split.")
    parser.add_argument("--config", type=str, default="configs/cinn_2d.yaml", help="Model config.")
    parser.add_argument("--eval-config", type=str, default="configs/eval.yaml", help="Evaluation config.")
    parser.add_argument("--smoke", action="store_true", help="Run fast evaluation with reduced bootstrap.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.eval_config, "r") as f:
        eval_cfg = yaml.safe_load(f)

    mode = cfg["data"]["mode"]
    splits_dir = cfg["data"]["splits_dir"]
    data_path = cfg["data"]["processed_pythia"]
    ckpt_dir = cfg["data"]["checkpoint_dir"]
    stage_num = "2" if mode == "2d" else "3"
    metrics_out = f"results/metrics/stage{stage_num}_pythia.json"
    fig_dir = "results/figures"
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.smoke else cfg["training"]["seeds"]
    n_bootstrap = 10 if args.smoke else eval_cfg["evaluation"]["num_bootstrap"]
    n_sbc_events = 200 if args.smoke else eval_cfg["evaluation"]["sbc"]["num_events"]
    n_sbc_samples = 50 if args.smoke else eval_cfg["evaluation"]["sbc"]["num_samples"]
    null_repeats = 5 if args.smoke else eval_cfg["evaluation"]["c2st"]["null_repeats"]

    # Load data
    logger.info("Loading Pythia %s dataset from %s...", mode, data_path)
    data = np.load(data_path)
    x_gen = data["x_gen"]
    y_sim = data["y_sim"]
    matched = data["matched"]
    target_names = list(data["target_names"])
    conditioning_names = list(data["conditioning_names"])
    dim_x = len(target_names)

    train_idx = np.load(os.path.join(splits_dir, "train_idx.npy"))
    test_idx = np.load(os.path.join(splits_dir, "test_idx.npy"))

    # Test matched events
    test_matched = test_idx[matched[test_idx]]
    train_matched = train_idx[matched[train_idx]]
    logger.info("Test matched events: %d (out of %d)", len(test_matched), len(test_idx))

    x_test_true = x_gen[test_matched]
    y_test_reco = y_sim[test_matched]

    # Load fitted pipelines
    pipe_x = Pipeline.load_from_checkpoint(os.path.join(ckpt_dir, f"transform_x_{mode}.pt"))
    pipe_y = Pipeline.load_from_checkpoint(os.path.join(ckpt_dir, f"transform_y_{mode}.pt"))

    # 1. Baseline: Reco-as-answer
    target_indices_in_y = [conditioning_names.index(t) for t in target_names]
    x_reco_answer = reco_as_answer(y_test_reco, target_indices_in_y)
    reco_ks = [ks_statistic(x_test_true[:, d], x_reco_answer[:, d]) for d in range(dim_x)]
    reco_w1 = [wasserstein_distance_1d(x_test_true[:, d], x_reco_answer[:, d]) for d in range(dim_x)]
    logger.info("Reco-as-answer | KS: %s | W1: %s", [f"{v:.4f}" for v in reco_ks], [f"{v:.4f}" for v in reco_w1])

    # 2. Baseline: Prior-only
    x_prior_answer = prior_only(x_gen[train_matched], num_samples=len(x_test_true), seed=42)
    prior_ks = [ks_statistic(x_test_true[:, d], x_prior_answer[:, d]) for d in range(dim_x)]
    prior_w1 = [wasserstein_distance_1d(x_test_true[:, d], x_prior_answer[:, d]) for d in range(dim_x)]
    prior_pair_c2st = pair_c2st(x_test_true, y_test_reco, x_prior_answer, null_repeats=null_repeats, seed=42)
    logger.info("Prior-only     | KS: %s | W1: %s | Pair C2ST: %.4f", [f"{v:.4f}" for v in prior_ks], [f"{v:.4f}" for v in prior_w1], prior_pair_c2st["auc"])

    # 3. Baseline: IBU (if 2D)
    ibu_results = None
    if mode == "2d":
        logger.info("Fitting and running 2D IBU baseline...")
        ibu = IBU2D(
            bins_per_dim=eval_cfg["evaluation"]["ibu"]["bins_per_dim"],
            num_iterations=eval_cfg["evaluation"]["ibu"]["num_iterations"],
        )
        ibu.fit_response(x_train_gen=x_gen, y_train_sim=y_sim, matched_train=matched)
        ibu_out = ibu.unfold(y_test_reco)

        # Binned chi2/ndf against truth for IBU marginals
        h_true_0, _ = np.histogram(x_test_true[:, 0], bins=ibu_out["edges_x0"])
        h_true_1, _ = np.histogram(x_test_true[:, 1], bins=ibu_out["edges_x1"])
        chi2_ibu_0, p_ibu_0, _ = binned_chi2_ndf(ibu_out["marginal_0"], h_true_0, err_pred=ibu_out["marginal_0_err"])
        chi2_ibu_1, p_ibu_1, _ = binned_chi2_ndf(ibu_out["marginal_1"], h_true_1, err_pred=ibu_out["marginal_1_err"])
        ibu_results = {
            "chi2_ndf": [chi2_ibu_0, chi2_ibu_1],
            "p_values": [p_ibu_0, p_ibu_1],
            "marginal_0": ibu_out["marginal_0"].tolist(),
            "marginal_0_err": ibu_out["marginal_0_err"].tolist(),
            "marginal_1": ibu_out["marginal_1"].tolist(),
            "marginal_1_err": ibu_out["marginal_1_err"].tolist(),
            "edges_x0": ibu_out["edges_x0"].tolist(),
            "edges_x1": ibu_out["edges_x1"].tolist(),
        }
        logger.info("IBU 2D         | Chi2/ndf: [%.2f, %.2f]", chi2_ibu_0, chi2_ibu_1)

    # 4. Evaluate cINN models across seeds
    cinn_seed_results = []
    unfolded_samples_per_seed = []
    latent_checks = []

    for seed in seeds:
        logger.info("Evaluating cINN (Seed %d)...", seed)
        cinn_model = build_model(cfg, unconditional=False)
        ckpt_path = os.path.join(ckpt_dir, f"cinn_{mode}_seed{seed}.pt")
        cinn_model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Latent space check
        with torch.no_grad():
            x_test_norm, _ = pipe_x(x_test_true, training=False)
            y_test_norm, _ = pipe_y(y_test_reco, training=False)
        lat_res = latent_check(cinn_model, x_test_norm.numpy(), y_test_norm.numpy(), device=device)
        latent_checks.append(lat_res)
        logger.info("Latent check seed %d: pass=%s (means=%s, stds=%s, max|corr|=%.3f)", seed, lat_res["overall_pass"], [f"{m:.3f}" for m in lat_res["means"]], [f"{s:.3f}" for s in lat_res["stds"]], lat_res["max_abs_offdiag_corr"])

        # Unfold test reco events (K=1)
        x_unf = unfold(cinn_model, y_test_reco, pipe_x, pipe_y, k=1, batch_size=eval_cfg["evaluation"]["batch_size"], device=device)
        unfolded_samples_per_seed.append(x_unf)

        ks_vals = [ks_statistic(x_test_true[:, d], x_unf[:, d]) for d in range(dim_x)]
        w1_vals = [wasserstein_distance_1d(x_test_true[:, d], x_unf[:, d]) for d in range(dim_x)]
        pair_res = pair_c2st(x_test_true, y_test_reco, x_unf, null_repeats=null_repeats, seed=seed)
        marg_res = marginal_c2st(x_test_true, x_unf, null_repeats=null_repeats, seed=seed)

        # Binned chi2/ndf bulk vs tails
        chi2_bulk_list = []
        for d in range(dim_x):
            q_bulk = np.quantile(x_test_true[:, d], np.linspace(0.05, 0.95, 21))
            h_tr, _ = np.histogram(x_test_true[:, d], bins=q_bulk)
            h_un, _ = np.histogram(x_unf[:, d], bins=q_bulk)
            c2_b, _, _ = binned_chi2_ndf(h_un, h_tr)
            chi2_bulk_list.append(c2_b)

        seed_data = {
            "seed": seed,
            "ks": ks_vals,
            "w1": w1_vals,
            "chi2_bulk": chi2_bulk_list,
            "pair_c2st": pair_res,
            "marginal_c2st": marg_res,
            "latent": lat_res,
        }
        cinn_seed_results.append(seed_data)
        logger.info("cINN Seed %d     | KS: %s | W1: %s | Pair C2ST: %.4f (Null thresh: %.4f)", seed, [f"{v:.4f}" for v in ks_vals], [f"{v:.4f}" for v in w1_vals], pair_res["auc"], pair_res["null_threshold"])

    # Aggregate across seeds
    primary_unf = unfolded_samples_per_seed[0]
    mean_w1 = np.mean([s["w1"] for s in cinn_seed_results], axis=0).tolist()
    std_w1 = np.std([s["w1"] for s in cinn_seed_results], axis=0).tolist()
    mean_ks = np.mean([s["ks"] for s in cinn_seed_results], axis=0).tolist()
    std_ks = np.std([s["ks"] for s in cinn_seed_results], axis=0).tolist()
    mean_pair_auc = float(np.mean([s["pair_c2st"]["auc"] for s in cinn_seed_results]))
    std_pair_auc = float(np.std([s["pair_c2st"]["auc"] for s in cinn_seed_results]))

    # 5. Evaluate Unconditional Control Flow
    logger.info("Evaluating Unconditional Control Model...")
    ctrl_model = build_model(cfg, unconditional=True)
    ctrl_ckpt = os.path.join(ckpt_dir, f"control_{mode}_seed0.pt")
    ctrl_model.load_state_dict(torch.load(ctrl_ckpt, map_location=device))

    x_ctrl = sample_unconditional(ctrl_model, pipe_x, num_samples=len(x_test_true), device=device)
    ctrl_ks = [ks_statistic(x_test_true[:, d], x_ctrl[:, d]) for d in range(dim_x)]
    ctrl_w1 = [wasserstein_distance_1d(x_test_true[:, d], x_ctrl[:, d]) for d in range(dim_x)]
    ctrl_pair_c2st = pair_c2st(x_test_true, y_test_reco, x_ctrl, null_repeats=null_repeats, seed=42)
    logger.info("Control Flow   | KS: %s | W1: %s | Pair C2ST: %.4f", [f"{v:.4f}" for v in ctrl_ks], [f"{v:.4f}" for v in ctrl_w1], ctrl_pair_c2st["auc"])

    # 6. SBC Rank Uniformity Test
    logger.info("Running SBC Rank Test (K=%d samples, M=%d events)...", n_sbc_samples, n_sbc_events)
    primary_cinn_model = build_model(cfg, unconditional=False)
    primary_cinn_model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"cinn_{mode}_seed0.pt"), map_location=device))
    with torch.no_grad():
        x_norm_all, _ = pipe_x(x_test_true, training=False)
        y_norm_all, _ = pipe_y(y_test_reco, training=False)
    sbc_results = sbc_rank_test(primary_cinn_model, x_norm_all.numpy(), y_norm_all.numpy(), k=n_sbc_samples, num_events=n_sbc_events, device=device)
    plot_sbc_ranks(sbc_results, target_names, os.path.join(fig_dir, f"sbc_ranks_{mode}"))

    # 7. Slice Tests
    logger.info("Running Slice Tests across 5 quantile slices...")
    slices_mass = slice_test(x_test_true, y_test_reco, primary_unf, slice_feature_idx=0, num_quantiles=5)

    # 8. Physical Constraints check (in 6D or 2D)
    constraints = {}
    if "mult" in target_names:
        idx_m = target_names.index("mult")
        neg_mult = int((primary_unf[:, idx_m] < 1.0).sum())
        non_int_mult = int((primary_unf[:, idx_m] % 1.0 != 0.0).sum())
        constraints["negative_or_zero_mult"] = neg_mult
        constraints["non_integer_mult"] = non_int_mult

    if "zg" in target_names:
        idx_zg = target_names.index("zg")
        out_zg = int(((primary_unf[:, idx_zg] < 0.0) | (primary_unf[:, idx_zg] > 0.5)).sum())
        constraints["zg_outside_0_0.5"] = out_zg

    if "sdm" in target_names and "mass" in target_names:
        idx_sdm = target_names.index("sdm")
        idx_mass = target_names.index("mass")
        viol = int((primary_unf[:, idx_sdm] > primary_unf[:, idx_mass] + 1e-3).sum())
        constraints["sdm_greater_than_mass_violations"] = viol
        constraints["sdm_greater_than_mass_fraction"] = float(viol / len(primary_unf))

    # 9. Correlation matrix check (especially 6D)
    corr_true = np.corrcoef(x_test_true, rowvar=False)
    corr_unf = np.corrcoef(primary_unf, rowvar=False)
    corr_diff = np.abs(corr_unf - corr_true)
    frob_norm = float(np.linalg.norm(corr_diff, "fro"))
    max_abs_corr_diff = float(np.max(corr_diff))
    corr_stats = {
        "frobenius_norm": frob_norm,
        "max_abs_diff": max_abs_corr_diff,
    }

    # 10. Generate Benchmark Figures
    logger.info("Generating benchmark comparison figures...")
    for d, name in enumerate(target_names):
        ibu_arg = None
        if mode == "2d" and ibu_results is not None:
            if d == 0:
                ibu_arg = (np.array(ibu_results["marginal_0"]), np.array(ibu_results["marginal_0_err"]), np.array(ibu_results["edges_x0"]))
            elif d == 1:
                ibu_arg = (np.array(ibu_results["marginal_1"]), np.array(ibu_results["marginal_1_err"]), np.array(ibu_results["edges_x1"]))

        plot_unfolding_marginal_comparison(
            x_true=x_test_true[:, d],
            x_reco=x_reco_answer[:, d],
            x_unfolded=primary_unf[:, d],
            obs_name=name,
            output_prefix=os.path.join(fig_dir, f"benchmark_{mode}_{name}"),
            x_ibu=ibu_arg,
        )

    if mode == "2d":
        plot_2d_contour_overlay(x_test_true, primary_unf, target_names, os.path.join(fig_dir, f"contour_2d_{target_names[0]}_{target_names[1]}"))
    elif mode == "6d":
        plot_corner_pairwise(x_test_true, primary_unf, target_names, os.path.join(fig_dir, "corner_6d_pairwise"))

    # Save all raw metrics to json
    final_metrics = {
        "mode": mode,
        "target_names": target_names,
        "reco_baseline": {"ks": reco_ks, "w1": reco_w1},
        "prior_baseline": {"ks": prior_ks, "w1": prior_w1, "pair_c2st": prior_pair_c2st},
        "control_baseline": {"ks": ctrl_ks, "w1": ctrl_w1, "pair_c2st": ctrl_pair_c2st},
        "ibu_baseline": ibu_results,
        "cinn_seeds": cinn_seed_results,
        "cinn_aggregate": {
            "w1_mean": mean_w1,
            "w1_std": std_w1,
            "ks_mean": mean_ks,
            "ks_std": std_ks,
            "pair_c2st_auc_mean": mean_pair_auc,
            "pair_c2st_auc_std": std_pair_auc,
        },
        "sbc_results": sbc_results,
        "slice_tests_mass": slices_mass,
        "constraints": constraints,
        "correlation_stats": corr_stats,
    }

    with open(metrics_out, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info("Saved all evaluation metrics to %s", metrics_out)


if __name__ == "__main__":
    main()
