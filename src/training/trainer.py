"""
Unified training orchestrator.

Runs the full training pipeline:
  1. Load synthetic data splits
  2. Preprocess (fit on train, apply to val/test)
  3. Extract features
  4. Optionally run stratified k-fold CV
  5. Train final models on full train+val
  6. Evaluate on held-out test set
  7. Save models and metrics

Usage:
    python -m src.training.trainer --dry-run     # 1 CV fold, quick validation
    python -m src.training.trainer               # Full 5-fold CV
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_config, setup_logging, resolve_path
from src.data.loader import DataLoader, CHANNEL_NAMES
from src.data.preprocessor import Preprocessor
from src.features.extractor import FeatureExtractor
from src.models.base_model import BaseModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network import NeuralNetworkModel
from src.evaluation.metrics import FaultMetrics
from src.training.cross_validator import cross_validate

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "neural_network": NeuralNetworkModel,
}


class Trainer:
    """Orchestrates end-to-end training and evaluation of all models."""

    def __init__(self, cfg: dict, dry_run: bool = False, model_name: Optional[str] = None):
        self.cfg = cfg
        self.dry_run = dry_run
        self.model_name = model_name
        self.seed = cfg["general"]["random_seed"]

        # Paths
        self.synthetic_dir = resolve_path(cfg, "synthetic_data")
        self.models_dir = resolve_path(cfg, "models_out")
        self.reports_dir = resolve_path(cfg, "reports_out")
        self.plots_dir = resolve_path(cfg, "plots_out")

        for d in [self.models_dir, self.reports_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple:
        """Load train/val/test splits. Falls back to synthetic generation if missing."""
        loader = DataLoader(self.cfg)
        train_csv = self.synthetic_dir / "train.csv"

        if not train_csv.exists():
            logger.warning("Synthetic data not found. Generating now...")
            from src.data.generator import DatasetGenerator
            gen = DatasetGenerator(self.cfg)
            df = gen.generate()
            gen.save(df, self.synthetic_dir)

        train_df, val_df, test_df = loader.load_splits(self.synthetic_dir)

        X_train_raw = train_df[CHANNEL_NAMES].values.astype(np.float32)
        y_train = train_df["label"].values.astype(np.int32)
        X_val_raw = val_df[CHANNEL_NAMES].values.astype(np.float32)
        y_val = val_df["label"].values.astype(np.int32)
        X_test_raw = test_df[CHANNEL_NAMES].values.astype(np.float32)
        y_test = test_df["label"].values.astype(np.int32)

        return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test

    # ------------------------------------------------------------------
    # Preprocessing and feature extraction
    # ------------------------------------------------------------------

    def _preprocess_and_extract(
        self,
        X_train_raw, y_train,
        X_val_raw, y_val,
        X_test_raw, y_test,
    ) -> tuple:
        """Apply preprocessing pipeline and extract feature windows."""
        preprocessor = Preprocessor(self.cfg)
        extractor = FeatureExtractor(self.cfg)

        logger.info("Fitting preprocessing pipeline on training data...")
        win_train, y_win_train, pipeline = preprocessor.fit_transform(X_train_raw, y_train)

        logger.info("Applying preprocessing to val/test splits...")
        win_val, y_win_val = preprocessor.transform(X_val_raw, y_val)
        win_test, y_win_test = preprocessor.transform(X_test_raw, y_test)

        logger.info("Extracting features from windows...")
        X_feat_train = extractor.transform(win_train)
        X_feat_val = extractor.transform(win_val)
        X_feat_test = extractor.transform(win_test)

        feature_names = extractor.get_feature_names()
        logger.info("Feature extraction complete: %d features", len(feature_names))

        # Save preprocessor pipeline
        pipeline_path = self.models_dir / "preprocessing_pipeline.joblib"
        preprocessor.save_pipeline(pipeline_path)

        return (
            X_feat_train, y_win_train,
            X_feat_val, y_win_val,
            X_feat_test, y_win_test,
            win_train, win_val, win_test,   # Raw windows for NN
            feature_names,
        )

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def _select_models(self) -> Dict[str, Type[BaseModel]]:
        if self.model_name:
            if self.model_name not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {self.model_name}. Choose from {list(MODEL_REGISTRY)}")
            return {self.model_name: MODEL_REGISTRY[self.model_name]}
        return MODEL_REGISTRY

    def _train_single_model(
        self,
        name: str,
        model_cls: Type[BaseModel],
        X_feat_train, y_train,
        X_feat_val, y_val,
        win_train, win_val,  # Raw windows for NN
    ) -> BaseModel:
        """Instantiate and train one model, using windows for NN and features for tree models."""
        model = model_cls(self.cfg)

        # NN operates on raw windows (channel-first); tree models on feature matrix
        if name == "neural_network":
            X_tr, X_vl = win_train, win_val
        else:
            X_tr, X_vl = X_feat_train, X_feat_val

        logger.info("Training %s...", name)
        t0 = time.time()
        train_metrics = model.train(X_tr, y_train, X_vl, y_val)
        elapsed = time.time() - t0
        logger.info("%s trained in %.1fs: %s", name, elapsed, train_metrics)
        return model

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _run_cv(
        self,
        model_cls: Type[BaseModel],
        X: np.ndarray,
        y: np.ndarray,
        name: str,
    ) -> Dict[str, float]:
        n_folds = 1 if self.dry_run else self.cfg["training"]["cv_folds"]
        logger.info("Running %d-fold CV for %s...", n_folds, name)
        cv_results = cross_validate(model_cls, self.cfg, X, y, n_splits=n_folds)
        return cv_results

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, dict]:
        """
        Execute the full training pipeline.

        Returns:
            Dict mapping model name → evaluation metrics on test set.
        """
        logger.info("=== Fault Prediction Training Pipeline ===")
        logger.info("Dry run: %s", self.dry_run)

        # Step 1: Load data
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = self._load_data()
        logger.info("Data loaded: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))

        # Step 2: Preprocess + extract features
        (
            X_feat_train, y_win_train,
            X_feat_val, y_win_val,
            X_feat_test, y_win_test,
            win_train, win_val, win_test,
            feature_names,
        ) = self._preprocess_and_extract(
            X_train_raw, y_train,
            X_val_raw, y_val,
            X_test_raw, y_test,
        )

        models_to_train = self._select_models()
        all_results = {}
        fm = FaultMetrics(self.cfg)

        for name, model_cls in models_to_train.items():
            logger.info("--- Model: %s ---", name)

            # Step 3: Cross-validation (uses feature matrix for all models for CV speed)
            cv_results = self._run_cv(model_cls, X_feat_train, y_win_train, name)

            # Step 4: Train final model
            model = self._train_single_model(
                name, model_cls,
                X_feat_train, y_win_train,
                X_feat_val, y_win_val,
                win_train, win_val,
            )

            # Step 5: Evaluate on test set
            X_test_input = win_test if name == "neural_network" else X_feat_test
            y_pred = model.predict(X_test_input)
            y_proba = model.predict_proba(X_test_input)
            test_metrics = fm.compute_all(y_win_test, y_pred, y_proba)
            test_metrics["cv_results"] = cv_results

            self._check_safety_gates(name, test_metrics)
            all_results[name] = test_metrics

            # Step 6: Save model
            model_path = self.models_dir / f"{name}.joblib"
            if name == "neural_network":
                model_path = self.models_dir / f"{name}.pt"
            model.save(model_path)

            # Step 7: Save feature importances if available
            importances = model.get_feature_importance()
            if importances is not None and len(importances) == len(feature_names):
                self._save_importance_csv(name, importances, feature_names)

        # Step 8: Save comparative metrics CSV
        self._save_metrics_csv(all_results)

        # Step 9: Generate evaluation report
        self._generate_report(all_results, feature_names)

        logger.info("=== Training pipeline complete ===")
        return all_results

    def _check_safety_gates(self, name: str, metrics: Dict) -> None:
        """Warn if safety-critical FDR/FAR thresholds are not met."""
        eval_cfg = self.cfg["evaluation"]
        fdr_gate = eval_cfg["dry_run_fdr_gate"] if self.dry_run else eval_cfg["min_fault_detection_rate"]
        far_gate = eval_cfg["dry_run_far_gate"] if self.dry_run else eval_cfg["max_false_alarm_rate"]

        fdr = metrics.get("fdr_mean", 0.0)
        far = metrics.get("far_mean", 0.0)

        if fdr < fdr_gate:
            logger.warning("[%s] SAFETY GATE: FDR %.3f < required %.3f", name, fdr, fdr_gate)
        else:
            logger.info("[%s] FDR gate PASSED: %.3f >= %.3f", name, fdr, fdr_gate)

        if far > far_gate:
            logger.warning("[%s] SAFETY GATE: FAR %.3f > allowed %.3f", name, far, far_gate)
        else:
            logger.info("[%s] FAR gate PASSED: %.3f <= %.3f", name, far, far_gate)

    def _save_importance_csv(
        self,
        model_name: str,
        importances: np.ndarray,
        feature_names: List[str],
    ) -> None:
        path = self.reports_dir / f"{model_name}_feature_importance.csv"
        pairs = sorted(zip(feature_names, importances.tolist()), key=lambda x: -x[1])
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feature", "importance"])
            writer.writerows(pairs)
        logger.info("Feature importance saved: %s", path)

    def _save_metrics_csv(self, all_results: Dict[str, dict]) -> None:
        path = self.reports_dir / "comparative_metrics.csv"
        scalar_keys = [
            k for k in list(all_results.values())[0].keys()
            if isinstance(list(all_results.values())[0][k], (int, float))
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + list(all_results.keys()))
            for key in scalar_keys:
                row = [key] + [f"{all_results[m].get(key, ''):.4f}" for m in all_results]
                writer.writerow(row)
        logger.info("Comparative metrics saved: %s", path)

    def _generate_report(self, all_results: Dict[str, dict], feature_names: List[str]) -> None:
        """Generate an HTML evaluation report."""
        from src.evaluation.explainability import generate_html_report
        try:
            report_path = self.reports_dir / "evaluation_report.html"
            generate_html_report(all_results, feature_names, report_path)
            logger.info("Evaluation report saved: %s", report_path)
        except Exception as e:
            logger.warning("Report generation failed: %s", e)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fault prediction training pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 CV fold for quick validation")
    parser.add_argument("--model", default=None, choices=list(MODEL_REGISTRY.keys()),
                        help="Train only a specific model")
    args = parser.parse_args()

    cfg = get_config(args.config)
    setup_logging(cfg["general"]["log_level"])

    trainer = Trainer(cfg, dry_run=args.dry_run, model_name=args.model)
    results = trainer.run()

    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<30s}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
