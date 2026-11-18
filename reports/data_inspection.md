# Data Inspection Report: Z+jets Delphes (OmniFold Dataset)

- Mode: Smoke (20k events)
- Pythia26 total events: 20,000
- Herwig total events: 20,000

## 1. Jet Matching and Unmatched Fractions

Events are selected with a hard Z boson ($p_T^Z > 200$ GeV). Delphes outputs the hardest jet at gen-level and at sim-level. Matching is determined by $\Delta R(j_{\text{gen}}, j_{\text{sim}}) \le 0.4$ and positive transverse momentum $p_T > 0$.

| Metric | Pythia26 | Herwig |
|---|---|---|
| Matched Fraction ($\Delta R \le 0.4$) | 97.9500% | 98.1550% |
| Unmatched / Fake / Miss Fraction | 2.0500% | 1.8450% |
| $\Delta R$ median (50%) | 0.00836 | 0.00876 |
| $\Delta R$ 95th percentile | 0.06248 | 0.05959 |
| $\Delta R$ 99th percentile | 1.66281 | 1.42454 |

## 2. Array Metadata and Statistics (Pythia26)

| Key | Shape | Dtype | Min | Max | Mean | Std | NaN / Inf | Exact Zero Frac | Discrete? |
|---|---|---|---|---|---|---|---|---|---|
| `gen_sdms` | (20000,) | `float64` | -8.314e-06 | 120.7 | 10.06 | 12.07 | 0 / 0 | 0.0100% | No |
| `gen_lhas` | (20000,) | `float64` | 0 | 0.7612 | 0.2925 | 0.1248 | 0 / 0 | 0.0050% | No |
| `gen_zgs` | (20000,) | `float64` | 0 | 0.5 | 0.233 | 0.1159 | 0 / 0 | 0.9700% | No |
| `gen_mults` | (20000,) | `int64` | 1 | 96 | 24.6 | 9.9 | 0 / 0 | 0.0000% | Yes |
| `gen_Zs` | (20000, 3) | `float32` | -3.686 | 1500 | 89.71 | 133 | 0 / 0 | 0.0000% | No |
| `gen_widths` | (20000,) | `float64` | 0 | 0.5954 | 0.1216 | 0.09325 | 0 / 0 | 0.0050% | No |
| `gen_tau2s` | (20000,) | `float64` | 0 | 0.3738 | 0.07684 | 0.04431 | 0 / 0 | 0.0250% | No |
| `gen_ang2s` | (20000,) | `float64` | 0 | 0.3929 | 0.04291 | 0.04733 | 0 / 0 | 0.0050% | No |
| `gen_jets` | (20000, 4) | `float64` | -3.771 | 1523 | 64.23 | 108 | 0 / 0 | 0.0000% | No |
| `sim_sdms` | (20000,) | `float64` | -2.107e-05 | 122 | 10.63 | 11.3 | 0 / 0 | 0.0650% | No |
| `sim_lhas` | (20000,) | `float64` | 0 | 0.7911 | 0.3231 | 0.1207 | 0 / 0 | 0.0200% | No |
| `sim_zgs` | (20000,) | `float64` | 0 | 0.4998 | 0.2396 | 0.1193 | 0 / 0 | 2.0850% | No |
| `sim_mults` | (20000,) | `int64` | 1 | 69 | 17.5 | 7.863 | 0 / 0 | 0.0000% | Yes |
| `sim_widths` | (20000,) | `float64` | 0 | 0.6469 | 0.1394 | 0.09371 | 0 / 0 | 0.0200% | No |
| `sim_tau2s` | (20000,) | `float64` | 0 | 0.4386 | 0.08573 | 0.05017 | 0 / 0 | 0.1000% | No |
| `sim_ang2s` | (20000,) | `float64` | 0 | 0.4585 | 0.04809 | 0.04923 | 0 / 0 | 0.0200% | No |
| `sim_jets` | (20000, 4) | `float64` | -3.87 | 1304 | 61.8 | 106.1 | 0 / 0 | 0.0000% | No |

## 3. Key Findings on Variables and Preprocessing Choices

1. **Single jet per event:** Each event stores exactly one hardest gen jet and one hardest sim jet (`shape: (N, 4)`).
2. **Exact zero point masses ($z_g$ and $m_{SD}$):**
   - `zgs`: Groomed momentum fraction has an exact zero point mass when soft drop fails to find a two-prong splitting. Handled via `ZeroInflatedLogitTransform` with uniform smoothing in training and threshold snapping at inference.
   - `sdms` and `mass`: Masses can have small numerical artifacts near zero ($< 0$), clamped cleanly to 0.0 before log-transform.
3. **Discrete variables:** `mults` (constituent multiplicity) is an integer count. Handled via `DequantizeTransform` with uniform noise $U(0, 1)$ during training and exact integer recovery via `floor`.
4. **Bounded variables:** $z_g \in [0, 0.5]$ and width/tau2 $\in [0, 1]$ handled via `Logit01Transform`.
5. **Figures generated:** Saved to `results/figures/inspect_pythia_*.png` and `.pdf`.
