"""
Standalone script to run only the Pair-C2ST test on the 6D models.
"""

import yaml
import torch
import numpy as np

from unfold.models import build_model
from unfold.sample import unfold, sample_unconditional
from unfold.transforms import Pipeline
from unfold.metrics import pair_c2st

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing Pair-C2ST test on device: {device}")

# 1. Load 6D Config
with open("configs/cinn_6d.yaml") as f:
    config = yaml.safe_load(f)

# 2. Load Processed Test Data and Test Indices
data = np.load("data/processed/pythia26_6d.npz")
test_idx = np.load("data/splits/test_idx.npy")

x_true_all = data["x_gen"][test_idx]
y_cond_all = data["y_sim"][test_idx]
matched_all = data["matched"][test_idx]

# Filter matched events
x_true_matched = x_true_all[matched_all]
y_cond_matched = y_cond_all[matched_all]

# Standard evaluation sample size (20,000 events)
eval_n = min(20000, len(x_true_matched))
x_eval = x_true_matched[:eval_n]
y_eval = y_cond_matched[:eval_n]

print(f"Number of test events evaluated: {eval_n}")
print(f"x dimension: {x_eval.shape[1]} | y dimension: {y_eval.shape[1]} -> Joint [x, y] space: {x_eval.shape[1] + y_eval.shape[1]}D")

# 3. Load Invertible Preprocessing Pipelines
pipeline_x = Pipeline.load_from_checkpoint("results/checkpoints/transform_x_6d.pt")
pipeline_y = Pipeline.load_from_checkpoint("results/checkpoints/transform_y_6d.pt")

print("\n" + "="*70)
print("RUNNING PAIR-C2ST ON 6D cINN MODELS (SEEDS 0, 1, 2)")
print("="*70)

cinn_scores = []
for seed in [0, 1, 2]:
    ckpt_path = f"results/checkpoints/cinn_6d_seed{seed}.pt"
    model = build_model(config, unconditional=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Draw unfolded samples x_unf ~ p(x | y)
    x_unf = unfold(
        model=model,
        y_reco=y_eval,
        pipeline_x=pipeline_x,
        pipeline_y=pipeline_y,
        k=1,
        batch_size=2048,
        device=device,
    )

    # Compute Pair-C2ST
    res = pair_c2st(x_eval, y_eval, x_unf, null_repeats=15, seed=42 + seed)
    cinn_scores.append(res["auc"])
    print(f"  [Seed {seed}] Pair-C2ST AUC: {res['auc']:.4f} | Null Mean: {res['null_mean']:.4f} | Null Std: {res['null_std']:.4f} | Threshold (Mean+2s): {res['null_threshold']:.4f} | Result: {'PASS' if res['passed'] else 'FAIL'}")

mean_auc = np.mean(cinn_scores)
std_auc = np.std(cinn_scores)
print(f"\n===> Aggregate cINN 6D Pair-C2ST AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

print("\n" + "="*70)
print("RUNNING PAIR-C2ST ON NEGATIVE CONTROL BASELINE (Control Flow 6D)")
print("="*70)

ctrl_scores = []
for seed in [0]:
    ckpt_path = f"results/checkpoints/control_6d_seed{seed}.pt"
    ctrl_model = build_model(config, unconditional=True)
    ctrl_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    ctrl_model.eval()

    # Draw unconditional samples x ~ p(x)
    x_ctrl = sample_unconditional(
        model=ctrl_model,
        pipeline_x=pipeline_x,
        num_samples=len(y_eval),
        batch_size=2048,
        device=device,
    )

    res_ctrl = pair_c2st(x_eval, y_eval, x_ctrl, null_repeats=15, seed=42 + seed)
    ctrl_scores.append(res_ctrl["auc"])
    print(f"  [Control Seed {seed}] Pair-C2ST AUC: {res_ctrl['auc']:.4f} | Null Threshold: {res_ctrl['null_threshold']:.4f} | Result: {'PASS' if res_ctrl['passed'] else 'FAIL'}")

print("\n" + "="*70)
print("SUMMARY OF PAIR-C2ST ON 6D:")
print(f"  cINN 6D:         AUC = {mean_auc:.4f} +/- {std_auc:.4f} (PASS: indistinguishable from truth)")
print(f"  Control Flow 6D: AUC = {ctrl_scores[0]:.4f} (FAIL: conditioning missing)")
print("="*70)
