"""
Feature selection and importance ranking.

Provides utilities to:
  - Rank features by importances from trained models
  - Remove highly correlated features
  - Select top-k features for dimensionality reduction
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects informative features based on model importances and correlation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit_from_importances(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        top_k: int = 60,
    ) -> "FeatureSelector":
        """
        Select top-k features by importance score.

        Args:
            importances: (n_features,) array of importance values.
            feature_names: List of feature name strings.
            top_k: Number of top features to retain.

        Returns:
            self
        """
        if len(importances) != len(feature_names):
            raise ValueError("importances and feature_names must have the same length")

        sorted_idx = np.argsort(importances)[::-1]
        self.selected_indices_ = sorted_idx[:top_k]
        self.feature_names_ = [feature_names[i] for i in self.selected_indices_]
        logger.info(
            "Selected %d features by importance. Top 5: %s",
            top_k,
            self.feature_names_[:5],
        )
        return self

    def fit_correlation_filter(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.95,
    ) -> "FeatureSelector":
        """
        Remove features with pairwise Pearson correlation above threshold.

        Iteratively removes the feature with higher mean correlation when
        a pair exceeds the threshold.

        Args:
            X: (n_samples, n_features) feature matrix.
            feature_names: Feature name list.
            threshold: Correlation threshold above which one feature is dropped.

        Returns:
            self
        """
        corr = np.corrcoef(X.T)
        n = len(feature_names)
        keep = list(range(n))

        for i in range(n):
            if i not in keep:
                continue
            for j in range(i + 1, n):
                if j not in keep:
                    continue
                if abs(corr[i, j]) > threshold:
                    # Drop feature with higher mean absolute correlation to others
                    mean_i = np.mean(np.abs(corr[i, keep]))
                    mean_j = np.mean(np.abs(corr[j, keep]))
                    drop = j if mean_i <= mean_j else i
                    keep.remove(drop)

        self.selected_indices_ = np.array(keep)
        self.feature_names_ = [feature_names[i] for i in keep]
        logger.info(
            "Correlation filter kept %d / %d features (threshold=%.2f)",
            len(keep), n, threshold,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply selection to feature matrix."""
        if self.selected_indices_ is None:
            raise RuntimeError("Call fit_from_importances or fit_correlation_filter first")
        return X[:, self.selected_indices_]

    def get_importance_df(
        self,
        importances: np.ndarray,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Return sorted DataFrame of feature names and importance scores."""
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
