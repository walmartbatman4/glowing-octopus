"""
Script 05: Evaluate prior-shift robustness on the unseen Herwig dataset.
Strictly NO retraining, tuning, or scaler re-fitting on Herwig.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml

from unfold.baselines import IBU2D, prior_only, reco_as_answer
from unfold.metrics import (
    binned_chi2_ndf,
    ks_statistic,
    marginal_c2st,
    pair_c2st,
    wasserstein_distance_1d,
)
from unfold.models import build_model
from unfold.plotting import plot_unfolding_marginal_comparison
from unfold.sample import unfold
from unfold.transforms import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("05_prior_shift")


def main():
    parser = argparse.ArgumentParser(description="Evaluate prior-shift on Herwig.")
    parser.add_argument("--config", type=str, default="configs/cinn_2d.yaml", help="Model config.")
    parser.add_argument("--eval-config", type=str, default="configs/eval.yaml", help="Evaluation config.")
    parser.add_argument("--smoke", action="store_true", help="Run fast evaluation with reduced sample.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.eval_config, "r") as f:
        eval_cfg = yaml.safe_load(f)

    mode = cfg["data"]["mode"]
    herwig_path = cfg["data"]["processed_herwig"]
    pythia_path = cfg["data"]["processed_pythia"]
    ckpt_dir = cfg["data"]["checkpoint_dir"]
    stage_num = "2" if mode == "2d" else "3"
    metrics_out = f"results/metrics/stage{stage_num}_herwig.json"
    fig_dir = "results/figures"
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    null_repeats = 5 if args.smoke else eval_cfg["evaluation"]["c2st"]["null_repeats"]

    # Load Herwig data
    logger.info("Loading Herwig %s dataset from %s...", mode, herwig_path)
    herwig_data = np.load(herwig_path)
    x_herwig_gen = herwig_data["x_gen"]
    y_herwig_sim = herwig_data["y_sim"]
    matched_herwig = herwig_data["matched"]
    target_names = list(herwig_data["target_names"])
    conditioning_names = list(herwig_data["conditioning_names"])
    dim_x = len(target_names)

    # Matched Herwig events for truth comparison
    matched_idx = np.where(matched_herwig)[0]
    if args.smoke:
        matched_idx = matched_idx[:3000]

    x_true_herwig = x_herwig_gen[matched_idx]
    y_reco_herwig = y_herwig_sim[matched_idx]
    logger.info("Evaluating on %d matched Herwig events...", len(x_true_herwig))

    # Load Pythia-fitted transforms
    pipe_x = Pipeline.load_from_checkpoint(os.path.join(ckpt_dir, f"transform_x_{mode}.pt"))
    pipe_y = Pipeline.load_from_checkpoint(os.path.join(ckpt_dir, f"transform_y_{mode}.pt"))

    # Load Pythia training data for prior-only baseline
    pythia_data = np.load(pythia_path)
    train_idx = np.load(os.path.join(cfg["data"]["splits_dir"], "train_idx.npy"))
    pythia_train_matched = train_idx[pythia_data["matched"][train_idx]]
    x_pythia_train = pythia_data["x_gen"][pythia_train_matched]

    # 1. Baseline: Reco-as-answer on Herwig
    target_indices_in_y = [conditioning_names.index(t) for t in target_names]
    x_reco_herwig = reco_as_answer(y_reco_herwig, target_indices_in_y)
    reco_ks = [ks_statistic(x_true_herwig[:, d], x_reco_herwig[:, d]) for d in range(dim_x)]
    reco_w1 = [wasserstein_distance_1d(x_true_herwig[:, d], x_reco_herwig[:, d]) for d in range(dim_x)]
    logger.info("Herwig Reco-as-answer | KS: %s | W1: %s", [f"{v:.4f}" for v in reco_ks], [f"{v:.4f}" for v in reco_w1])

    # 2. Baseline: Prior-only (Pythia truth prior applied to Herwig truth)
    x_prior_answer = prior_only(x_pythia_train, num_samples=len(x_true_herwig), seed=42)
    prior_ks = [ks_statistic(x_true_herwig[:, d], x_prior_answer[:, d]) for d in range(dim_x)]
    prior_w1 = [wasserstein_distance_1d(x_true_herwig[:, d], x_prior_answer[:, d]) for d in range(dim_x)]
    prior_pair_c2st = pair_c2st(x_true_herwig, y_reco_herwig, x_prior_answer, null_repeats=null_repeats, seed=42)
    logger.info("Herwig Prior-only     | KS: %s | W1: %s | Pair C2ST: %.4f", [f"{v:.4f}" for v in prior_ks], [f"{v:.4f}" for v in prior_w1], prior_pair_c2st["auc"])

    # 3. Baseline: IBU on Herwig (Pythia response matrix applied to Herwig sim)
    ibu_results = None
    if mode == "2d":
        logger.info("Running Pythia-trained IBU response matrix on Herwig sim...")
        ibu = IBU2D(
            bins_per_dim=eval_cfg["evaluation"]["ibu"]["bins_per_dim"],
            num_iterations=eval_cfg["evaluation"]["ibu"]["num_iterations"],
        )
        ibu.fit_response(x_train_gen=pythia_data["x_gen"], y_train_sim=pythia_data["y_sim"], matched_train=pythia_data["matched"])
        ibu_out = ibu.unfold(y_reco_herwig)

        h_true_0, _ = np.histogram(x_true_herwig[:, 0], bins=ibu_out["edges_x0"])
        h_true_1, _ = np.histogram(x_true_herwig[:, 1], bins=ibu_out["edges_x1"])
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
        logger.info("Herwig IBU 2D         | Chi2/ndf: [%.2f, %.2f]", chi2_ibu_0, chi2_ibu_1)

    # 4. Unfold Herwig sim with Pythia-trained cINN model (Seed 0)
    logger.info("Unfolding Herwig with Pythia-trained cINN (Seed 0)...")
    cinn_model = build_model(cfg, unconditional=False)
    cinn_model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"cinn_{mode}_seed0.pt"), map_location=device))

    x_unf_herwig = unfold(
        cinn_model,
        y_reco_herwig,
        pipe_x,
        pipe_y,
        k=1,
        batch_size=eval_cfg["evaluation"]["batch_size"],
        device=device,
    )

    cinn_ks = [ks_statistic(x_true_herwig[:, d], x_unf_herwig[:, d]) for d in range(dim_x)]
    cinn_w1 = [wasserstein_distance_1d(x_true_herwig[:, d], x_unf_herwig[:, d]) for d in range(dim_x)]
    cinn_pair_c2st = pair_c2st(x_true_herwig, y_reco_herwig, x_unf_herwig, null_repeats=null_repeats, seed=42)
    cinn_marg_c2st = marginal_c2st(x_true_herwig, x_unf_herwig, null_repeats=null_repeats, seed=42)
    logger.info("Herwig cINN           | KS: %s | W1: %s | Pair C2ST: %.4f (Null thresh: %.4f)", [f"{v:.4f}" for v in cinn_ks], [f"{v:.4f}" for v in cinn_w1], cinn_pair_c2st["auc"], cinn_pair_c2st["null_threshold"])

    # Prior-shift bias comparison table per observable: cINN vs IBU vs Prior-only
    bias_table = {}
    for d, name in enumerate(target_names):
        mean_diff_cinn = float(np.mean(x_unf_herwig[:, d]) - np.mean(x_true_herwig[:, d]))
        mean_diff_prior = float(np.mean(x_prior_answer[:, d]) - np.mean(x_true_herwig[:, d]))
        mean_diff_reco = float(np.mean(x_reco_herwig[:, d]) - np.mean(x_true_herwig[:, d]))
        bias_table[name] = {
            "w1_cinn": cinn_w1[d],
            "w1_prior": prior_w1[d],
            "w1_reco": reco_w1[d],
            "bias_mean_cinn": mean_diff_cinn,
            "bias_mean_prior": mean_diff_prior,
            "bias_mean_reco": mean_diff_reco,
        }

    # Generate benchmark figures on Herwig
    logger.info("Generating Herwig prior-shift comparison figures...")
    for d, name in enumerate(target_names):
        ibu_arg = None
        if mode == "2d" and ibu_results is not None:
            if d == 0:
                ibu_arg = (np.array(ibu_results["marginal_0"]), np.array(ibu_results["marginal_0_err"]), np.array(ibu_results["edges_x0"]))
            elif d == 1:
                ibu_arg = (np.array(ibu_results["marginal_1"]), np.array(ibu_results["marginal_1_err"]), np.array(ibu_results["edges_x1"]))

        plot_unfolding_marginal_comparison(
            x_true=x_true_herwig[:, d],
            x_reco=x_reco_herwig[:, d],
            x_unfolded=x_unf_herwig[:, d],
            obs_name=name,
            output_prefix=os.path.join(fig_dir, f"herwig_shift_{mode}_{name}"),
            x_ibu=ibu_arg,
        )

    herwig_metrics = {
        "mode": mode,
        "target_names": target_names,
        "reco_baseline": {"ks": reco_ks, "w1": reco_w1},
        "prior_baseline": {"ks": prior_ks, "w1": prior_w1, "pair_c2st": prior_pair_c2st},
        "ibu_baseline": ibu_results,
        "cinn": {
            "ks": cinn_ks,
            "w1": cinn_w1,
            "pair_c2st": cinn_pair_c2st,
            "marginal_c2st": cinn_marg_c2st,
        },
        "bias_table": bias_table,
    }

    with open(metrics_out, "w") as f:
        json.dump(herwig_metrics, f, indent=2)
    logger.info("Saved prior-shift metrics to %s", metrics_out)


if __name__ == "__main__":
    main()
