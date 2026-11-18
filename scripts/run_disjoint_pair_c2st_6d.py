"""
Pair-C2ST test on 6D where:
1. The conditioning dataset y is split into two disjoint halves: y_half1 and y_half2.
2. Class 1: [x_true_half1, y_half1] (True events on half 1).
3. Class 0: [x_unfold_half2, y_half2] (cINN unfolded events conditioned on half 2).
4. Train/test split (50/50) of this dataset on LightGBM and evaluate ROC-AUC.
"""

import yaml
import torch
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from unfold.models import build_model
from unfold.sample import unfold, sample_unconditional
from unfold.transforms import Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Load config & data
with open("configs/cinn_6d.yaml") as f:
    config = yaml.safe_load(f)

data = np.load("data/processed/pythia26_6d.npz")
test_idx = np.load("data/splits/test_idx.npy")

x_all = data["x_gen"][test_idx]
y_all = data["y_sim"][test_idx]
matched = data["matched"][test_idx]

x_matched = x_all[matched]
y_matched = y_all[matched]

# Use matched events
n_total = len(x_matched)
half = n_total // 2

# Split y (and corresponding truth) into two distinct halves
x_half1 = x_matched[:half]
y_half1 = y_matched[:half]

x_half2 = x_matched[half : 2 * half]
y_half2 = y_matched[half : 2 * half]

print(f"Total events: {n_total}")
print(f"Half 1 size (for x_true):   {len(y_half1)}")
print(f"Half 2 size (for x_unfold): {len(y_half2)}")

pipeline_x = Pipeline.load_from_checkpoint("results/checkpoints/transform_x_6d.pt")
pipeline_y = Pipeline.load_from_checkpoint("results/checkpoints/transform_y_6d.pt")

# Construct Class 1: [x_true, y_half1]
class1_pairs = np.concatenate([x_half1, y_half1], axis=-1)

# Function to run classifier test
def run_c2st_eval(p_true, p_fake, seed=42):
    X = np.concatenate([p_true, p_fake], axis=0)
    Y = np.concatenate([np.ones(len(p_true), dtype=int), np.zeros(len(p_fake), dtype=int)], axis=0)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=seed, stratify=Y
    )
    
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1
    )
    clf.fit(X_train, Y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(Y_test, probs))

# Null test: Class 1 = [x_true_half1, y_half1], Class 0 = [x_true_half2, y_half2]
null_auc = run_c2st_eval(class1_pairs, np.concatenate([x_half2, y_half2], axis=-1), seed=42)
print(f"\n[Baseline Null AUC: True Half 1 vs True Half 2]: {null_auc:.4f}")

print("\n" + "="*70)
print("EVALUATING DISJOINT HALF SPLIT PAIR-C2ST ON 6D cINN (SEEDS 0, 1, 2)")
print("="*70)

c2st_results = []
for seed in [0, 1, 2]:
    ckpt = f"results/checkpoints/cinn_6d_seed{seed}.pt"
    model = build_model(config, unconditional=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    # Generate x_unfold strictly conditioned on y_half2
    x_unfold_half2 = unfold(
        model=model,
        y_reco=y_half2,
        pipeline_x=pipeline_x,
        pipeline_y=pipeline_y,
        k=1,
        batch_size=2048,
        device=device,
    )
    
    # Construct Class 0: [x_unfold_half2, y_half2]
    class0_pairs = np.concatenate([x_unfold_half2, y_half2], axis=-1)
    
    auc = run_c2st_eval(class1_pairs, class0_pairs, seed=42 + seed)
    c2st_results.append(auc)
    print(f"  [cINN 6D Seed {seed}] Pair-C2ST AUC: {auc:.4f} (Null reference: {null_auc:.4f})")

print(f"\n===> Mean cINN 6D Pair-C2ST AUC: {np.mean(c2st_results):.4f} +/- {np.std(c2st_results):.4f}")

# Negative Control (Control Flow without conditioning)
print("\n" + "="*70)
print("EVALUATING DISJOINT HALF SPLIT ON CONTROL FLOW (NO CONDITIONING)")
print("="*70)
ctrl_model = build_model(config, unconditional=True)
ctrl_model.load_state_dict(torch.load("results/checkpoints/control_6d_seed0.pt", map_location=device))
ctrl_model.eval()

x_ctrl_half2 = sample_unconditional(
    model=ctrl_model,
    pipeline_x=pipeline_x,
    num_samples=len(y_half2),
    batch_size=2048,
    device=device,
)
class0_ctrl_pairs = np.concatenate([x_ctrl_half2, y_half2], axis=-1)
auc_ctrl = run_c2st_eval(class1_pairs, class0_ctrl_pairs, seed=42)
print(f"  [Control Flow 6D Seed 0] Pair-C2ST AUC: {auc_ctrl:.4f}")
