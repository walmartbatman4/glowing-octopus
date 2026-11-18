# Stage 1 Acceptance Report: Data Download and Preprocessing

## Summary of Acceptance Checks

| Item | Requirement | Result | Details |
|---|---|---|---|
| 1.1 | `pytest tests/test_transforms.py` passes | **PASS** | 7 tests passed (exact roundtrip, Jacobian autograd equivalence, dequantization integer recovery, zero-inflated snapping, standardize). |
| 1.2 | `reports/data_inspection.md` exists with all plots | **PASS** | Inspection report generated with per-key metadata, zero fractions, matching statistics, and 10 distribution figures saved as PNG and PDF in `results/figures/`. |
| 1.3 | Splits are disjoint and complete; Herwig isolated | **PASS** | Verified via `tests/test_splits.py`. Train (70%), Val (15%), Test (15%) are strictly disjoint and cover all events. Herwig data is strictly reserved for prior-shift evaluation with no splits or scaler fitting. |

## Data Properties Identified
- **Dataset**: Z+jets Delphes fast simulation (Zenodo record 3548091).
- **Matching**: Jet radius $R=0.4$. Hardest gen and sim jets matched at $\Delta R \le 0.4$ with $p_T > 0$. Unmatched fraction: 2.05% in Pythia26, 1.85% in Herwig.
- **Hard variables handled**:
  - $z_g$: Point mass at 0.0 handled via `ZeroInflatedLogitTransform` with uniform training smoothing and inference threshold snapping.
  - Multiplicity: Integer count handled via `DequantizeTransform` with uniform noise $U(0, 1)$ in training and floor recovery.
  - Bounded observables ($z_g, \text{width}, \tau_2$): Bounded logit mappings.
  - Mass and $m_{SD}$: Clamped to $\ge 0$ and log-transformed.
