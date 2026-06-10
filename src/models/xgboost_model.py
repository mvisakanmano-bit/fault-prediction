"""
XGBoost classifier for power system fault detection.

Key design choices:
  - Early stopping on validation mlogloss (50 rounds patience)
  - scale_pos_weight computed from training class distribution
  - SHAP TreeExplainer for per-prediction feature contributions
  - RandomizedSearchCV for hyperparameter search
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost multi-class classifier with early stopping and SHAP explainability.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost>=2.0.0")
        self._importances: Optional[np.ndarray] = None
        self._shap_explainer: Optional[object] = None
        xgb_cfg = cfg["models"]["xgboost"]
        self._early_stop = xgb_cfg["early_stopping_rounds"]
        self._eval_metric = xgb_cfg["eval_metric"]
        self._search_iter = xgb_cfg["random_search_iter"]
        self._cv_folds = xgb_cfg["cv_folds"]
        self._seed = cfg["general"]["random_seed"]
        self._n_classes = cfg["data_generation"]["num_classes"]

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        For multi-class XGBoost, scale_pos_weight as ratio of
        negative to positive samples for the majority class.
        Used as a proxy for overall class imbalance weight.
        """
        counts = np.bincount(y.astype(int))
        majority = counts.max()
        minority = counts.min()
        return float(majority / max(minority, 1))

    def _param_grid(self, scale_pos_weight: float) -> dict:
        xgb_cfg = self.cfg["models"]["xgboost"]
        return {
            "n_estimators": xgb_cfg["n_estimators_options"],
            "max_depth": xgb_cfg["max_depth_options"],
            "learning_rate": xgb_cfg["learning_rate_options"],
            "subsample": xgb_cfg["subsample_options"],
            "colsample_bytree": xgb_cfg["colsample_bytree_options"],
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """
        Hyperparameter search then final fit with early stopping.

        Early stopping uses the validation set to prevent overfitting
        and reduce unnecessary training time.
        """
        logger.info("XGBoost: starting hyperparameter search (%d iterations)...", self._search_iter)

        spw = self._compute_scale_pos_weight(y_train)

        base_xgb = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=self._n_classes,
            eval_metric=self._eval_metric,
            random_state=self._seed,
            n_jobs=-1,
            verbosity=0,
        )

        cv = StratifiedKFold(n_splits=self._cv_folds, shuffle=True, random_state=self._seed)
        search = RandomizedSearchCV(
            base_xgb,
            param_distributions=self._param_grid(spw),
            n_iter=self._search_iter,
            cv=cv,
            scoring="f1_macro",
            random_state=self._seed,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_params = search.best_params_
        logger.info("Best XGBoost params: %s | CV F1=%.4f", best_params, search.best_score_)

        # Final fit with early stopping on validation set
        self.model = xgb.XGBClassifier(
            **best_params,
            objective="multi:softprob",
            num_class=self._n_classes,
            eval_metric=self._eval_metric,
            random_state=self._seed,
            n_jobs=-1,
            verbosity=1,
            early_stopping_rounds=self._early_stop,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # Build SHAP explainer (TreeExplainer is fast for tree models)
        if SHAP_AVAILABLE:
            try:
                self._shap_explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP TreeExplainer built successfully")
            except Exception as e:
                logger.warning("SHAP explainer failed: %s", e)

        # Store feature importances (weight-based from XGBoost)
        self._importances = self.model.feature_importances_

        self.is_fitted = True

        val_preds = self.model.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        logger.info("XGBoost: val_accuracy=%.4f, best_iteration=%d", val_acc, self.model.best_iteration)

        return {
            "val_accuracy": val_acc,
            "best_cv_f1": float(search.best_score_),
            "best_iteration": int(self.model.best_iteration),
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

    def get_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute SHAP values for input samples.

        Returns:
            SHAP values array of shape (n_samples, n_features, n_classes)
            or None if SHAP unavailable.
        """
        if self._shap_explainer is None:
            return None
        try:
            return self._shap_explainer.shap_values(X)
        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "importances": self._importances,
            # SHAP explainer is rebuilt from model on load
        }
        joblib.dump(payload, path)
        logger.info("XGBoost saved to %s", path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self._importances = payload["importances"]
        if SHAP_AVAILABLE:
            try:
                self._shap_explainer = shap.TreeExplainer(self.model)
            except Exception:
                pass
        self.is_fitted = True
        logger.info("XGBoost loaded from %s", path)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        return self._importances
