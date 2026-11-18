"""
Feature extraction and observable management for unfolding.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_single_observable(
    raw_data: Dict[str, np.ndarray],
    obs_name: str,
    level: str = "gen",
) -> np.ndarray:
    """Extract a single observable array for either gen or sim level.

    Args:
        raw_data: Dictionary containing raw arrays.
        obs_name: Observable identifier ('mass', 'pt', 'eta', 'phi', 'mult',
                  'width', 'sdm', 'zg', 'tau2', 'lha', 'ang2').
        level: 'gen' or 'sim'.

    Returns:
        1D array of shape (N,) with the observable values.
    """
    prefix = f"{level}_"

    if obs_name == "mass":
        arr = raw_data[f"{prefix}jets"][:, 3]
        return np.maximum(arr, 0.0)
    elif obs_name == "pt":
        return raw_data[f"{prefix}jets"][:, 0]
    elif obs_name == "eta" or obs_name == "y":
        return raw_data[f"{prefix}jets"][:, 1]
    elif obs_name == "phi":
        return raw_data[f"{prefix}jets"][:, 2]
    elif obs_name == "mult":
        return raw_data[f"{prefix}mults"].astype(np.float64)
    elif obs_name == "width":
        return raw_data[f"{prefix}widths"]
    elif obs_name == "sdm":
        arr = raw_data[f"{prefix}sdms"]
        return np.maximum(arr, 0.0)
    elif obs_name == "zg":
        arr = raw_data[f"{prefix}zgs"]
        return np.clip(arr, 0.0, 0.5)
    elif obs_name == "tau2":
        return raw_data[f"{prefix}tau2s"]
    elif obs_name == "lha":
        return raw_data[f"{prefix}lhas"]
    elif obs_name == "ang2":
        return raw_data[f"{prefix}ang2s"]
    else:
        raise ValueError(f"Unknown observable name '{obs_name}'")


def build_feature_arrays(
    raw_data: Dict[str, np.ndarray],
    target_names: List[str],
    conditioning_names: List[str],
    dr_max: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build truth target x_gen, reco conditioning y_sim, matched mask, and weights.

    Args:
        raw_data: Raw dictionary of Delphes arrays.
        target_names: List of observable names for x_gen.
        conditioning_names: List of observable names for y_sim.
        dr_max: Delta R matching threshold.

    Returns:
        x_gen: shape (N, Dx)
        y_sim: shape (N, Dy)
        matched: shape (N,) bool
        weights: shape (N,) float64 (all 1.0 by default)
    """
    from unfold.io import extract_matched_flags

    _, matched = extract_matched_flags(raw_data, dr_max=dr_max)
    n_events = len(matched)

    x_cols = [
        extract_single_observable(raw_data, name, level="gen")
        for name in target_names
    ]
    y_cols = [
        extract_single_observable(raw_data, name, level="sim")
        for name in conditioning_names
    ]

    x_gen = np.stack(x_cols, axis=-1)
    y_sim = np.stack(y_cols, axis=-1)
    weights = np.ones(n_events, dtype=np.float64)

    return x_gen, y_sim, matched, weights
