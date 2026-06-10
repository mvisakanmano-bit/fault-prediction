"""
Abstract base class for all fault prediction models.

Every model must implement this interface to be usable by the
training loop, evaluation pipeline, and inference engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class BaseModel(ABC):
    """
    Abstract interface for fault classification models.

    Subclasses: RandomForestModel, XGBoostModel, NeuralNetworkModel.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model: Any = None          # Underlying estimator
        self.is_fitted: bool = False
        self.class_names: List[str] = [
            "Normal", "SLG", "LL", "3PH", "Overload", "VoltageSag", "Harmonic"
        ]

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fit model on training data, use validation set for early stopping or tuning.

        Args:
            X_train: (n_train, n_features) training features.
            y_train: (n_train,) integer class labels.
            X_val:   (n_val, n_features) validation features.
            y_val:   (n_val,) integer class labels.

        Returns:
            Dict of training metrics (e.g., {"val_accuracy": 0.97}).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: (n_samples, n_features) feature array.

        Returns:
            (n_samples,) integer predictions.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities. Required for alarm thresholds and calibration.

        Args:
            X: (n_samples, n_features) feature array.

        Returns:
            (n_samples, n_classes) probability matrix, rows sum to 1.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Return feature importance array (n_features,) or None if unavailable.

        For RF/XGBoost: impurity-based or SHAP importances.
        For NN: gradient-based saliency or None.
        """
        ...

    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__
