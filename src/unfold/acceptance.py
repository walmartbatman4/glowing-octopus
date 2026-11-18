"""
Acceptance, efficiency, and purity corrections using binary classifiers.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


class AcceptanceCorrection:
    """Classifier-based efficiency eps(x) and purity 1 - p_fake(y) correction."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        min_efficiency: float = 0.05,
        max_weight: float = 20.0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_efficiency = min_efficiency
        self.max_weight = max_weight
        self.eff_model: Optional[lgb.LGBMClassifier] = None
        self.purity_model: Optional[lgb.LGBMClassifier] = None

    def fit_efficiency(self, x_gen_all: np.ndarray, matched: np.ndarray, seed: int = 42) -> None:
        """Fit P(matched | x_gen)."""
        logger.info("Training efficiency classifier on %d total gen events...", len(x_gen_all))
        clf = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        clf.fit(x_gen_all, matched.astype(int))
        self.eff_model = clf

    def fit_purity(self, y_sim_all: np.ndarray, matched: np.ndarray, seed: int = 42) -> None:
        """Fit P(matched | y_sim) = 1 - P(fake | y_sim)."""
        logger.info("Training purity classifier on %d total sim events...", len(y_sim_all))
        clf = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        clf.fit(y_sim_all, matched.astype(int))
        self.purity_model = clf

    def get_efficiency_weights(self, x_gen: np.ndarray) -> np.ndarray:
        """Compute unfolding event weights 1 / eps(x)."""
        if self.eff_model is None:
            return np.ones(len(x_gen))
        eps = self.eff_model.predict_proba(x_gen)[:, 1]
        eps = np.clip(eps, self.min_efficiency, 1.0)
        weights = 1.0 / eps
        return np.clip(weights, 0.0, self.max_weight)

    def get_purity_weights(self, y_sim: np.ndarray) -> np.ndarray:
        """Compute reco event weights: P(matched | y_sim)."""
        if self.purity_model is None:
            return np.ones(len(y_sim))
        purity = self.purity_model.predict_proba(y_sim)[:, 1]
        return np.clip(purity, 0.0, 1.0)
