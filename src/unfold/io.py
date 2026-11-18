"""
Input/Output utilities for downloading, loading, and matching Z+jets Delphes data.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DELPHES_KEYS = [
    "jets",
    "Zs",
    "ang2s",
    "lhas",
    "mults",
    "sdms",
    "tau2s",
    "widths",
    "zgs",
]


def compute_delta_r(
    y1: np.ndarray,
    phi1: np.ndarray,
    y2: np.ndarray,
    phi2: np.ndarray,
) -> np.ndarray:
    """Compute Delta R between two sets of coordinates."""
    dy = y1 - y2
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)
    return np.sqrt(dy**2 + dphi**2)


def load_raw_dataset(
    dataset: str = "Pythia26",
    num_data: Optional[int] = None,
    cache_dir: str = "data/raw",
    source: str = "zenodo",
) -> Dict[str, np.ndarray]:
    """Load raw dataset using energyflow.zjets_delphes with fallbacks.

    Args:
        dataset: "Pythia26" or "Herwig" (or "Pythia21", "Pythia25").
        num_data: Number of events to load (None or -1 for all).
        cache_dir: Local directory for cached files.
        source: Primary source ('zenodo' or 'dropbox').

    Returns:
        Dictionary of numpy arrays for gen and sim quantities.
    """
    import energyflow.datasets.zjets_delphes as zd

    n_data = -1 if num_data is None else int(num_data)
    os.makedirs(cache_dir, exist_ok=True)

    sources_to_try = [source]
    if source == "zenodo":
        sources_to_try.append("dropbox")
    elif source == "dropbox":
        sources_to_try.append("zenodo")

    data = None
    for src in sources_to_try:
        try:
            logger.info("Attempting to load %s from %s...", dataset, src)
            data = zd.load(
                dataset,
                num_data=n_data,
                pad=False,
                cache_dir=cache_dir,
                source=src,
                exclude_keys=["particles"],
            )
            logger.info("Successfully loaded %s from %s (%d events).", dataset, src, len(data["gen_jets"]))
            break
        except Exception as e:
            logger.warning("Failed to load %s from %s: %s", dataset, src, e)

    if data is None:
        logger.info("Trying zenodo_get fallback...")
        try:
            subprocess.run(["zenodo_get", "3548091", "-o", cache_dir], check=True)
            data = zd.load(
                dataset,
                num_data=n_data,
                pad=False,
                cache_dir=cache_dir,
                source="zenodo",
                exclude_keys=["particles"],
            )
        except Exception as e:
            raise RuntimeError(f"All download attempts failed for {dataset}: {e}") from e

    return data


def extract_matched_flags(
    raw_data: Dict[str, np.ndarray],
    dr_max: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Delta R and boolean matched mask between hardest gen and sim jets.

    Args:
        raw_data: Dictionary containing 'gen_jets' and 'sim_jets' [pt, y, phi, m].
        dr_max: Maximum Delta R for matching.

    Returns:
        dr: Delta R per event array of shape (N,).
        matched: Boolean array of shape (N,) where True indicates a valid matched pair.
    """
    gen_jets = raw_data["gen_jets"]
    sim_jets = raw_data["sim_jets"]

    dr = compute_delta_r(
        y1=gen_jets[:, 1],
        phi1=gen_jets[:, 2],
        y2=sim_jets[:, 1],
        phi2=sim_jets[:, 2],
    )

    valid_pt = (gen_jets[:, 0] > 0.0) & (sim_jets[:, 0] > 0.0)
    matched = (dr <= dr_max) & valid_pt
    return dr, matched
