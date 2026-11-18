"""
Script 01: Download and inspect Z+jets Delphes dataset (Pythia26 and Herwig).
Generates exploratory plots and writes reports/data_inspection.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

from unfold.features import extract_single_observable
from unfold.io import extract_matched_flags, load_raw_dataset
from unfold.plotting import plot_observable_distribution

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("01_download_inspect")


def inspect_dataset(data: Dict[str, np.ndarray], name: str) -> Dict[str, Any]:
    """Compute detailed summary statistics for a dataset."""
    stats = {}
    dr, matched = extract_matched_flags(data, dr_max=0.4)
    n_total = len(matched)
    n_matched = int(matched.sum())

    stats["total_events"] = n_total
    stats["matched_events"] = n_matched
    stats["matched_fraction"] = float(matched.mean())
    stats["unmatched_fraction"] = float(1.0 - matched.mean())
    stats["dr_quantiles"] = {
        "q50": float(np.quantile(dr, 0.50)),
        "q90": float(np.quantile(dr, 0.90)),
        "q95": float(np.quantile(dr, 0.95)),
        "q99": float(np.quantile(dr, 0.99)),
    }

    key_stats = {}
    for key, arr in data.items():
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        zero_count = int((arr == 0).sum())
        zero_fraction = float(zero_count / arr.size)

        key_stats[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_fraction": zero_fraction,
            "is_discrete": (arr.dtype.kind in ("i", "u")) or (len(np.unique(arr[:1000])) < 100 and arr.ndim == 1),
        }
    stats["keys"] = key_stats
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and inspect Delphes dataset.")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode (20,000 events).")
    parser.add_argument("--num-data", type=int, default=None, help="Number of events to load.")
    parser.add_argument("--cache-dir", type=str, default="data/raw", help="Directory for raw data cache.")
    parser.add_argument("--fig-dir", type=str, default="results/figures", help="Directory for output figures.")
    parser.add_argument("--report-path", type=str, default="reports/data_inspection.md", help="Inspection report path.")
    args = parser.parse_args()

    num_data = 20000 if args.smoke else args.num_data
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    logger.info("Loading Pythia26 dataset (num_data=%s)...", num_data)
    pythia_data = load_raw_dataset("Pythia26", num_data=num_data, cache_dir=args.cache_dir)
    logger.info("Inspecting Pythia26 dataset...")
    pythia_stats = inspect_dataset(pythia_data, "Pythia26")

    logger.info("Loading Herwig dataset (num_data=%s)...", num_data)
    herwig_data = load_raw_dataset("Herwig", num_data=num_data, cache_dir=args.cache_dir)
    logger.info("Inspecting Herwig dataset...")
    herwig_stats = inspect_dataset(herwig_data, "Herwig")

    # Generate exploratory distribution plots for all candidate observables
    observables = ["mass", "mult", "width", "sdm", "zg", "tau2", "pt", "eta", "lha", "ang2"]
    logger.info("Generating candidate observable histograms...")
    for obs in observables:
        gen_vals = extract_single_observable(pythia_data, obs, level="gen")
        sim_vals = extract_single_observable(pythia_data, obs, level="sim")
        fig_prefix = os.path.join(args.fig_dir, f"inspect_pythia_{obs}")
        plot_observable_distribution(gen_vals, sim_vals, obs, fig_prefix)

    # Write reports/data_inspection.md
    logger.info("Writing inspection report to %s...", args.report_path)
    with open(args.report_path, "w") as f:
        f.write("# Data Inspection Report: Z+jets Delphes (OmniFold Dataset)\n\n")
        f.write(f"- Mode: {'Smoke (20k events)' if args.smoke else 'Full dataset'}\n")
        f.write(f"- Pythia26 total events: {pythia_stats['total_events']:,}\n")
        f.write(f"- Herwig total events: {herwig_stats['total_events']:,}\n\n")

        f.write("## 1. Jet Matching and Unmatched Fractions\n\n")
        f.write(
            "Events are selected with a hard Z boson ($p_T^Z > 200$ GeV). Delphes outputs the hardest jet at "
            "gen-level and at sim-level. Matching is determined by $\\Delta R(j_{\\text{gen}}, j_{\\text{sim}}) \\le 0.4$ "
            "and positive transverse momentum $p_T > 0$.\n\n"
        )
        f.write(f"| Metric | Pythia26 | Herwig |\n|---|---|---|\n")
        f.write(f"| Matched Fraction ($\\Delta R \\le 0.4$) | {pythia_stats['matched_fraction']:.4%} | {herwig_stats['matched_fraction']:.4%} |\n")
        f.write(f"| Unmatched / Fake / Miss Fraction | {pythia_stats['unmatched_fraction']:.4%} | {herwig_stats['unmatched_fraction']:.4%} |\n")
        f.write(f"| $\\Delta R$ median (50%) | {pythia_stats['dr_quantiles']['q50']:.5f} | {herwig_stats['dr_quantiles']['q50']:.5f} |\n")
        f.write(f"| $\\Delta R$ 95th percentile | {pythia_stats['dr_quantiles']['q95']:.5f} | {herwig_stats['dr_quantiles']['q95']:.5f} |\n")
        f.write(f"| $\\Delta R$ 99th percentile | {pythia_stats['dr_quantiles']['q99']:.5f} | {herwig_stats['dr_quantiles']['q99']:.5f} |\n\n")

        f.write("## 2. Array Metadata and Statistics (Pythia26)\n\n")
        f.write("| Key | Shape | Dtype | Min | Max | Mean | Std | NaN / Inf | Exact Zero Frac | Discrete? |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for k, v in pythia_stats["keys"].items():
            shape_str = str(tuple(v["shape"]))
            nan_inf_str = f"{v['nan_count']} / {v['inf_count']}"
            disc_str = "Yes" if v["is_discrete"] else "No"
            f.write(
                f"| `{k}` | {shape_str} | `{v['dtype']}` | {v['min']:.4g} | {v['max']:.4g} | "
                f"{v['mean']:.4g} | {v['std']:.4g} | {nan_inf_str} | {v['zero_fraction']:.4%} | {disc_str} |\n"
            )

        f.write("\n## 3. Key Findings on Variables and Preprocessing Choices\n\n")
        f.write("1. **Single jet per event:** Each event stores exactly one hardest gen jet and one hardest sim jet (`shape: (N, 4)`).\n")
        f.write("2. **Exact zero point masses ($z_g$ and $m_{SD}$):**\n")
        f.write("   - `zgs`: Groomed momentum fraction has an exact zero point mass when soft drop fails to find a two-prong splitting. Handled via `ZeroInflatedLogitTransform` with uniform smoothing in training and threshold snapping at inference.\n")
        f.write("   - `sdms` and `mass`: Masses can have small numerical artifacts near zero ($< 0$), clamped cleanly to 0.0 before log-transform.\n")
        f.write("3. **Discrete variables:** `mults` (constituent multiplicity) is an integer count. Handled via `DequantizeTransform` with uniform noise $U(0, 1)$ during training and exact integer recovery via `floor`.\n")
        f.write("4. **Bounded variables:** $z_g \\in [0, 0.5]$ and width/tau2 $\\in [0, 1]$ handled via `Logit01Transform`.\n")
        f.write("5. **Figures generated:** Saved to `results/figures/inspect_pythia_*.png` and `.pdf`.\n")

    logger.info("Data inspection complete!")


if __name__ == "__main__":
    main()
