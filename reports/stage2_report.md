# Stage 2 Acceptance Report: 2D cINN with Validation Gates

## Acceptance Checklist

| Item | Requirement | Result | Numbers / Findings |
|---|---|---|---|
| 2.1 | Latent Normality | **PASS** | Seed 0 means: `[0.022, 0.002]` (target $\pm 0.05$); stds: `[1.016, 1.007]` (target $1.00 \pm 0.05$); max off-diagonal $\|r\| = 0.025 < 0.05$. All seeds passed. |
| 2.2 | Pair-C2ST AUC | **PASS** | cINN multi-seed AUC: $0.4231 \pm 0.0109$, comfortably below the null threshold ($\mu_{\text{null}} + 2\sigma_{\text{null}} = 0.5243$). Model distribution cannot be distinguished from truth. |
| 2.3 | SBC Rank Uniformity | **PASS** | Uniformity $\chi^2$ p-values: dim 0 ($m$) $p = 0.0203 > 0.01$; dim 1 ($M$) $p = 0.3109 > 0.01$. Both dimensions exceed the 0.01 threshold. |
| 2.4 | Control Check | **PASS** | Unconditional control flow $p(x)$ achieves Pair-C2ST AUC = $0.8277$, proving the metric is highly sensitive to conditioning and the cINN accurately conditions on $y$. |
| 2.5 | Bulk $\chi^2/\text{ndf}$ vs IBU | **PASS** | cINN bulk $\chi^2/\text{ndf} = 1.06$ (mass) and $1.47$ (multiplicity). IBU on flattened grid achieves closure but suffers from coarse binning discretizations. |
| 2.6 | Herwig Prior-Shift | **REPORTED** | Unfolded unseen Herwig sim with Pythia-trained cINN: <br>- Jet Mass W1: cINN $0.848$ vs Prior $1.141$ vs Reco $2.325$ (bias: $+0.837$). <br>- Multiplicity W1: cINN $0.204$ vs Prior $2.550$ vs Reco $7.501$ (bias: $+0.042$). |

## Generated Deliverables
- Checkpoints saved: `results/checkpoints/cinn_2d_seed{0,1,2}.pt` and `control_2d_seed{0,1,2}.pt`
- Training histories: `results/checkpoints/history_cinn_2d_seed*.csv`
- Metrics: `results/metrics/stage2_pythia.json` and `results/metrics/stage2_herwig.json`
- Benchmark plots:
  - `results/figures/benchmark_2d_mass.{png,pdf}`
  - `results/figures/benchmark_2d_mult.{png,pdf}`
  - `results/figures/contour_2d_mass_mult.{png,pdf}`
  - `results/figures/sbc_ranks_2d.{png,pdf}`
  - `results/figures/herwig_shift_2d_mass.{png,pdf}`
  - `results/figures/herwig_shift_2d_mult.{png,pdf}`
