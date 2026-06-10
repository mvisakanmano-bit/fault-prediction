"""
Stratified k-fold cross-validation utilities.

Provides a cross_validate function that runs any BaseModel through
k folds and aggregates metrics, returning mean ± std for reporting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.base_model import BaseModel
from src.evaluation.metrics import FaultMetrics

logger = logging.getLogger(__name__)


def cross_validate(
    model_cls: Type[BaseModel],
    cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Stratified k-fold cross-validation for a model class.

    Each fold: instantiates a fresh model, trains on fold train split,
    evaluates on fold val split, collects all metrics.

    Args:
        model_cls: Uninstantiated BaseModel subclass.
        cfg: Full project config.
        X: Feature matrix (n_samples, n_features).
        y: Label vector (n_samples,).
        n_splits: Number of folds.

    Returns:
        Dict with "metric_mean" and "metric_std" for each metric key.
    """
    seed = cfg["general"]["random_seed"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        logger.info("Cross-validation fold %d / %d", fold_idx, n_splits)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_cls(cfg)
        model.train(X_train, y_train, X_val, y_val)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        fm = FaultMetrics(cfg)
        metrics = fm.compute_all(y_val, y_pred, y_proba)
        fold_metrics.append(metrics)
        logger.info("Fold %d metrics: acc=%.4f, f1_macro=%.4f", fold_idx, metrics["accuracy"], metrics["f1_macro"])

    # Aggregate: mean ± std across folds
    all_keys = fold_metrics[0].keys()
    aggregated = {}
    for key in all_keys:
        vals = [m[key] for m in fold_metrics if isinstance(m[key], (int, float))]
        if vals:
            aggregated[f"{key}_mean"] = float(np.mean(vals))
            aggregated[f"{key}_std"] = float(np.std(vals))

    logger.info(
        "CV complete — acc=%.4f±%.4f, f1_macro=%.4f±%.4f",
        aggregated.get("accuracy_mean", 0),
        aggregated.get("accuracy_std", 0),
        aggregated.get("f1_macro_mean", 0),
        aggregated.get("f1_macro_std", 0),
    )
    return aggregated
