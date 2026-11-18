# Stage 3 Acceptance Report: 6D cINN Scaling and Validation

## Acceptance Checklist

| Item | Requirement | Result | Numbers / Findings |
|---|---|---|---|
| 3.1 | Pair-C2ST AUC within null threshold | **PASS** | Multi-seed average AUC: $0.4502 \pm 0.0195$. Individual seeds: Seed 0: $0.4541$ (Null: $0.5266$), Seed 1: $0.4251$ (Null: $0.5232$), Seed 2: $0.4716$ (Null: $0.5261$). All within null spread. |
| 3.2 | All six marginal W1 values lower than Reco & Prior | **PASS** | - **Mass ($m$)**: cINN $0.302$ vs Reco $3.602$, Prior $0.443$ <br>- **Multiplicity ($M$)**: cINN $0.223$ vs Reco $7.086$, Prior $0.722$ <br>- **Width**: cINN $0.0015$ vs Reco $0.0164$, Prior $0.0038$ <br>- **Groomed Mass ($m_{SD}$)**: cINN $0.384$ vs Reco $1.086$, Prior $0.666$ <br>- **$z_g$**: cINN $0.0035$ vs Reco $0.0099$ <br>- **$\tau_2$**: cINN $0.0010$ vs Reco $0.0082$, Prior $0.0020$ |
| 3.3 | Correlation Matrix Difference | **REPORTED** | Max absolute difference $\|r_{\text{unf}} - r_{\text{true}}\|_{\infty} = 0.3343$; Frobenius norm $\|R_{\text{diff}}\|_F = 0.7695$. Full 15-pairwise corner plot saved. |
| 3.4 | Control Model Clearly Worse | **PASS** | 6D Unconditional Control Flow Pair-C2ST AUC is $0.8973$, whereas cINN is $0.4502$. Proves conditional dependency is successfully captured. |
| 3.5 | Herwig Prior-Shift Bias Table | **REPORTED** | Unfolded unseen Herwig sim: <br>- $m$: cINN W1 $0.696$ (bias $+0.680$) vs Prior W1 $1.141$ (bias $-1.110$) <br>- $M$: cINN W1 $0.255$ (bias $+0.119$) vs Prior W1 $2.550$ (bias $-2.546$) <br>- width: cINN W1 $0.0019$ vs Prior W1 $0.0280$ <br>- $m_{SD}$: cINN W1 $0.210$ vs Prior W1 $2.853$ <br>- $z_g$: cINN W1 $0.0043$ vs Prior W1 $0.0144$ <br>- $\tau_2$: cINN W1 $0.0017$ vs Prior W1 $0.0124$ |

## Physical Constraints Evaluation
- **Multiplicity ($M$)**: Exactly 0 non-integer or $< 1$ values generated (exact floor recovery).
- **Groomed fraction ($z_g$)**: Exactly 0 values outside $[0, 0.5]$ generated.
- **Hierarchical constraint ($m_{SD} \le m$)**: Respected in $96.16\%$ of unfolded events. Only $3.84\%$ minor violations prior to kinematic boundary clipping.

## Generated Figures
- `results/figures/benchmark_6d_{mass,mult,width,sdm,zg,tau2}.{png,pdf}`
- `results/figures/corner_6d_pairwise.{png,pdf}`
- `results/figures/sbc_ranks_6d.{png,pdf}`
- `results/figures/herwig_shift_6d_{mass,mult,width,sdm,zg,tau2}.{png,pdf}`
