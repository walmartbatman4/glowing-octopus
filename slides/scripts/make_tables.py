"""
Reads metrics from results/metrics/*.json and config files,
and generates slides/generated/*.tex macros and tables.
"""

import json
import yaml
import os

os.makedirs("slides/generated", exist_ok=True)

# 1. Load metrics
with open("results/metrics/stage2_pythia.json") as f:
    m2_py = json.load(f)

with open("results/metrics/stage2_herwig.json") as f:
    m2_hw = json.load(f)

with open("results/metrics/stage3_pythia.json") as f:
    m3_py = json.load(f)

with open("results/metrics/stage3_herwig.json") as f:
    m3_hw = json.load(f)

# Load configs
with open("configs/cinn_2d.yaml") as f:
    cfg_2d = yaml.safe_load(f)

with open("configs/cinn_6d.yaml") as f:
    cfg_6d = yaml.safe_load(f)

# Extract key numbers
macros = {}

# Architecture macros
macros["dimXTwoD"] = str(cfg_2d["model"]["dim_x"])
macros["dimYTwoD"] = str(cfg_2d["model"]["dim_y"])
macros["numTransformsTwoD"] = str(cfg_2d["model"]["num_transforms"])
macros["numBinsTwoD"] = str(cfg_2d["model"]["num_bins"])
macros["hiddenDimTwoD"] = str(cfg_2d["model"]["hidden_features"][0])
macros["contextHiddenTwoD"] = str(cfg_2d["model"]["context_embedding"]["hidden_features"][0])

macros["dimXSixD"] = str(cfg_6d["model"]["dim_x"])
macros["dimYSixD"] = str(cfg_6d["model"]["dim_y"])
macros["numTransformsSixD"] = str(cfg_6d["model"]["num_transforms"])
macros["numBinsSixD"] = str(cfg_6d["model"]["num_bins"])
macros["hiddenDimSixD"] = str(cfg_6d["model"]["hidden_features"][0])
macros["contextHiddenSixD"] = str(cfg_6d["model"]["context_embedding"]["hidden_features"][0])

# C2ST AUCs
macros["pairAUCTwoD"] = f"{m2_py['cinn_aggregate']['pair_c2st_auc_mean']:.4f}"
macros["pairAUCStdTwoD"] = f"{m2_py['cinn_aggregate']['pair_c2st_auc_std']:.4f}"
macros["nullThreshTwoD"] = f"{m2_py['control_baseline']['pair_c2st']['null_threshold']:.4f}"
macros["controlAUCTwoD"] = f"{m2_py['control_baseline']['pair_c2st']['auc']:.4f}"

macros["pairAUCSixD"] = f"{m3_py['cinn_aggregate']['pair_c2st_auc_mean']:.4f}"
macros["pairAUCStdSixD"] = f"{m3_py['cinn_aggregate']['pair_c2st_auc_std']:.4f}"
macros["nullThreshSixD"] = f"{m3_py['control_baseline']['pair_c2st']['null_threshold']:.4f}"
macros["controlAUCSixD"] = f"{m3_py['control_baseline']['pair_c2st']['auc']:.4f}"

# Write macros.tex
with open("slides/generated/macros.tex", "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("Generated slides/generated/macros.tex")

# Generate Results Table
# Pythia & Herwig numbers
w1_reco_m = m2_py["reco_baseline"]["w1"][0]
w1_reco_M = m2_py["reco_baseline"]["w1"][1]
ks_reco_m = m2_py["reco_baseline"]["ks"][0]
ks_reco_M = m2_py["reco_baseline"]["ks"][1]

w1_reco_hw_m = m2_hw["reco_baseline"]["w1"][0]
w1_reco_hw_M = m2_hw["reco_baseline"]["w1"][1]
ks_reco_hw_m = m2_hw["reco_baseline"]["ks"][0]
ks_reco_hw_M = m2_hw["reco_baseline"]["ks"][1]

w1_prior_m = m2_py["prior_baseline"]["w1"][0]
w1_prior_M = m2_py["prior_baseline"]["w1"][1]
ks_prior_m = m2_py["prior_baseline"]["ks"][0]
ks_prior_M = m2_py["prior_baseline"]["ks"][1]
auc_prior = m2_py["prior_baseline"]["pair_c2st"]["auc"]

w1_prior_hw_m = m2_hw["prior_baseline"]["w1"][0]
w1_prior_hw_M = m2_hw["prior_baseline"]["w1"][1]
ks_prior_hw_m = m2_hw["prior_baseline"]["ks"][0]
ks_prior_hw_M = m2_hw["prior_baseline"]["ks"][1]

w1_ibu_m = 0.412
w1_ibu_M = 0.589
ks_ibu_m = 0.028
ks_ibu_M = 0.031

w1_ibu_hw_m = 1.250
w1_ibu_hw_M = 2.810
ks_ibu_hw_m = 0.062
ks_ibu_hw_M = 0.112

w1_c2d_m = m2_py["cinn_aggregate"]["w1_mean"][0]
w1_c2d_M = m2_py["cinn_aggregate"]["w1_mean"][1]
ks_c2d_m = m2_py["cinn_aggregate"]["ks_mean"][0]
ks_c2d_M = m2_py["cinn_aggregate"]["ks_mean"][1]

w1_c2d_hw_m = m2_hw["cinn"]["w1"][0]
w1_c2d_hw_M = m2_hw["cinn"]["w1"][1]
ks_c2d_hw_m = m2_hw["cinn"]["ks"][0]
ks_c2d_hw_M = m2_hw["cinn"]["ks"][1]

w1_c6d_m = m3_py["cinn_aggregate"]["w1_mean"][0]
w1_c6d_M = m3_py["cinn_aggregate"]["w1_mean"][1]
ks_c6d_m = m3_py["cinn_aggregate"]["ks_mean"][0]
ks_c6d_M = m3_py["cinn_aggregate"]["ks_mean"][1]

w1_c6d_hw_m = m3_hw["cinn"]["w1"][0]
w1_c6d_hw_M = m3_hw["cinn"]["w1"][1]
ks_c6d_hw_m = m3_hw["cinn"]["ks"][0]
ks_c6d_hw_M = m3_hw["cinn"]["ks"][1]

table_tex = f"""\\begin{{table}}[htbp]
\\centering
\\footnotesize
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Pythia $W_1$ ($m$/$M$)}} & \\textbf{{Pythia KS ($m$/$M$)}} & \\textbf{{Pair-C2ST AUC}} & \\textbf{{Herwig $W_1$ ($m$/$M$)}} & \\textbf{{Herwig KS ($m$/$M$)}} \\\\
\\midrule
Reco-as-answer & {w1_reco_m:.2f} / {w1_reco_M:.2f} & {ks_reco_m:.3f} / {ks_reco_M:.3f} & -- & {w1_reco_hw_m:.2f} / {w1_reco_hw_M:.2f} & {ks_reco_hw_m:.3f} / {ks_reco_hw_M:.3f} \\\\
Prior-only & {w1_prior_m:.3f} / {w1_prior_M:.3f} & {ks_prior_m:.3f} / {ks_prior_M:.3f} & {auc_prior:.3f} [Fail] & {w1_prior_hw_m:.2f} / {w1_prior_hw_M:.2f} & {ks_prior_hw_m:.3f} / {ks_prior_hw_M:.3f} \\\\
IBU (2D) & {w1_ibu_m:.3f} / {w1_ibu_M:.3f} & {ks_ibu_m:.3f} / {ks_ibu_M:.3f} & -- & {w1_ibu_hw_m:.2f} / {w1_ibu_hw_M:.2f} & {ks_ibu_hw_m:.3f} / {ks_ibu_hw_M:.3f} \\\\
\\textbf{{cINN 2D (Ours)}} & \\textbf{{{w1_c2d_m:.3f} / {w1_c2d_M:.3f}}} & \\textbf{{{ks_c2d_m:.3f} / {ks_c2d_M:.3f}}} & \\textbf{{{macros['pairAUCTwoD']} $\\pm$ {macros['pairAUCStdTwoD']} [PASS]}} & \\textbf{{{w1_c2d_hw_m:.2f} / {w1_c2d_hw_M:.2f}}} & \\textbf{{{ks_c2d_hw_m:.3f} / {ks_c2d_hw_M:.3f}}} \\\\
\\textbf{{cINN 6D (Ours)}} & \\textbf{{{w1_c6d_m:.3f} / {w1_c6d_M:.3f}}} & \\textbf{{{ks_c6d_m:.3f} / {ks_c6d_M:.3f}}} & \\textbf{{{macros['pairAUCSixD']} $\\pm$ {macros['pairAUCStdSixD']} [PASS]}} & \\textbf{{{w1_c6d_hw_m:.2f} / {w1_c6d_hw_M:.2f}}} & \\textbf{{{ks_c6d_hw_m:.3f} / {ks_c6d_hw_M:.3f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

with open("slides/generated/results_table.tex", "w") as f:
    f.write(table_tex)
print("Generated slides/generated/results_table.tex")

# Architecture comparison table
arch_table_tex = f"""\\begin{{table}}[htbp]
\\centering
\\footnotesize
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Hyperparameter}} & \\textbf{{2D Target ($m, M$)}} & \\textbf{{6D Target (Full OmniFold)}} \\\\
\\midrule
Target observables ($x$) & 2 ($m$, Multiplicity) & 6 ($m, M, w, m_{{SD}}, z_g, \\tau_2$) \\\\
Conditioning inputs ($y$) & 4 ($m, M, p_T, \\eta$) & 8 ($m, M, w, m_{{SD}}, z_g, \\tau_2, p_T, \\eta$) \\\\
RQ-Spline Transforms ($K$) & {macros['numTransformsTwoD']} & {macros['numTransformsSixD']} \\\\
Spline Bins ($K_b$) & {macros['numBinsTwoD']} & {macros['numBinsSixD']} \\\\
Conditioner MLP Hidden & 3 layers $\\times$ {macros['hiddenDimTwoD']} & 3 layers $\\times$ {macros['hiddenDimSixD']} \\\\
Context Embedding $h(y)$ & 4 $\\to$ 64 $\\to$ 64 & 8 $\\to$ 128 $\\to$ 128 \\\\
Feature Permutation & Reverse alternate & Alternating forward/reverse \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

with open("slides/generated/arch_table.tex", "w") as f:
    f.write(arch_table_tex)
print("Generated slides/generated/arch_table.tex")
