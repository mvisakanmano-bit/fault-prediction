"""
Random Forest classifier for power system fault detection.

Key design choices:
  - RandomizedSearchCV for efficient hyperparameter search
  - class_weight='balanced' mandatory to handle class imbalance
  - CalibratedClassifierCV(isotonic) for well-calibrated fault probabilities
    (critical: alarm thresholds require trustworthy probability estimates)
  - n_jobs=-1 for full parallelism during training and inference
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest with randomized hyperparameter search and isotonic calibration.

    After hyperparameter search, the best estimator is wrapped in
    CalibratedClassifierCV so that predict_proba outputs reliable
    fault probability estimates (not just relative scores).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._base_model: Optional[RandomForestClassifier] = None
        self._importances: Optional[np.ndarray] = None
        rf_cfg = cfg["models"]["random_forest"]
        self._n_jobs = rf_cfg["n_jobs"]
        self._calibration_method = rf_cfg["calibration_method"]
        self._search_iter = rf_cfg["random_search_iter"]
        self._cv_folds = rf_cfg["cv_folds"]
        self._seed = cfg["general"]["random_seed"]

    def _param_grid(self) -> dict:
        rf_cfg = self.cfg["models"]["random_forest"]
        return {
            "n_estimators": rf_cfg["n_estimators_options"],
            "max_depth": rf_cfg["max_depth_options"],          # None already handled by yaml
            "min_samples_split": rf_cfg["min_samples_split_options"],
            "max_features": rf_cfg["max_features_options"],
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """
        Perform RandomizedSearchCV then calibrate the best estimator.

        The validation set is passed to calibration fitting so calibration
        is done on held-out data (avoids overconfidence from training-set calibration).
        """
        logger.info("RandomForest: starting hyperparameter search (%d iterations)...", self._search_iter)

        base_rf = RandomForestClassifier(
            class_weight="balanced",
            n_jobs=self._n_jobs,
            random_state=self._seed,
        )

        cv = StratifiedKFold(n_splits=self._cv_folds, shuffle=True, random_state=self._seed)
        search = RandomizedSearchCV(
            base_rf,
            param_distributions=self._param_grid(),
            n_iter=self._search_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=self._n_jobs,
            random_state=self._seed,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_params = search.best_params_
        logger.info("Best RF params: %s | CV F1=%.4f", best_params, search.best_score_)

        # Re-instantiate best estimator for calibration via cross-val on training data.
        # cv="prefit" was removed in sklearn 1.9; we use 3-fold cross-val calibration
        # on the training set instead, which remains robust and avoids API breakage.
        self._base_model = RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            n_jobs=self._n_jobs,
            random_state=self._seed,
        )
        cv_cal = StratifiedKFold(n_splits=3, shuffle=True, random_state=self._seed)
        self.model = CalibratedClassifierCV(
            self._base_model, method=self._calibration_method, cv=cv_cal
        )
        # Fit on combined train+val so calibration sees more data
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self.model.fit(X_full, y_full)

        # Extract importances from one of the calibrated sub-estimators.
        # self._base_model is cloned internally by sklearn's CV calibration,
        # so we must read from calibrated_classifiers_[0].estimator instead.
        sub = self.model.calibrated_classifiers_[0].estimator
        self._importances = sub.feature_importances_

        self.is_fitted = True

        val_preds = self.model.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        logger.info("RandomForest: val_accuracy=%.4f", val_acc)

        return {
            "val_accuracy": val_acc,
            "best_cv_f1": float(search.best_score_),
            "best_params": str(best_params),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "importances": self._importances}, path)
        logger.info("RandomForest saved to %s", path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self._importances = payload["importances"]
        self.is_fitted = True
        logger.info("RandomForest loaded from %s", path)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        return self._importances
