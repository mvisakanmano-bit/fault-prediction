"""
Single entry point for the fault prediction pipeline.

Usage:
    python run_pipeline.py --config config/config.yaml
    python run_pipeline.py --config config/config.yaml --model random_forest
    python run_pipeline.py --config config/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_config, setup_logging
from src.training.trainer import Trainer, MODEL_REGISTRY

import logging
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Fault Prediction Pipeline — end-to-end training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config config/config.yaml
  python run_pipeline.py --config config/config.yaml --model xgboost
  python run_pipeline.py --config config/config.yaml --dry-run
  python run_pipeline.py --config config/config.yaml --skip-generation
        """,
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--model", default=None, choices=list(MODEL_REGISTRY.keys()),
        help="Train only a specific model (default: all models)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 CV fold for quick validation (does not retrain final models)",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip synthetic data generation (use existing data in data/synthetic/)",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate synthetic data, then exit",
    )
    parser.add_argument(
        "--plot-explainability", action="store_true",
        help="Generate SHAP and Grad-CAM plots after training",
    )
    return parser.parse_args()


def run_data_generation(cfg: dict) -> None:
    """Phase 1: Generate and validate synthetic dataset."""
    from src.data.generator import DatasetGenerator
    from src.utils import resolve_path

    output_dir = resolve_path(cfg, "synthetic_data")
    if (output_dir / "train.csv").exists():
        logger.info("Synthetic data already exists at %s — skipping generation", output_dir)
        return

    logger.info("=== Phase 1: Synthetic Dataset Generation ===")
    gen = DatasetGenerator(cfg)
    df = gen.generate()
    gen.print_summary(df)

    ok = gen.validate(df)
    if not ok:
        logger.error("Dataset validation FAILED — aborting pipeline")
        sys.exit(1)

    gen.save(df, output_dir)
    gen.save_sample_plots(df, output_dir)
    logger.info("Phase 1 complete: dataset saved to %s", output_dir)


def main() -> None:
    args = parse_args()

    cfg = get_config(args.config)
    setup_logging(cfg["general"]["log_level"])

    logger.info("=" * 60)
    logger.info("AI Fault Prediction Pipeline")
    logger.info("Config: %s | Model: %s | Dry-run: %s",
                args.config, args.model or "all", args.dry_run)
    logger.info("=" * 60)

    # Phase 1: Data generation
    if not args.skip_generation:
        run_data_generation(cfg)

    if args.generate_only:
        logger.info("--generate-only flag set: stopping after data generation")
        return

    # Phase 2–5: Training pipeline
    logger.info("=== Phases 2–5: Preprocessing, Feature Extraction, Training, Evaluation ===")
    trainer = Trainer(cfg, dry_run=args.dry_run, model_name=args.model)
    results = trainer.run()

    # Optional: explainability plots
    if args.plot_explainability and results:
        logger.info("=== Phase 5b: Generating Explainability Plots ===")
        _generate_explainability_plots(cfg, results)

    # Final summary
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE — FINAL RESULTS")
    print("=" * 65)
    for model_name, metrics in results.items():
        print(f"\n  {model_name.upper()}")
        summary_keys = ["accuracy", "f1_macro", "mcc", "fdr_mean", "far_mean", "mttd_ms", "ece"]
        for k in summary_keys:
            v = metrics.get(k, "N/A")
            if isinstance(v, float):
                print(f"    {k:<30s}: {v:.4f}")
    print("=" * 65)

    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_out"]
    print(f"\n  Reports saved to: {reports_dir}")
    print(f"  Plots saved to:   {PROJECT_ROOT / cfg['paths']['plots_out']}")
    print(f"  Models saved to:  {PROJECT_ROOT / cfg['paths']['models_out']}")


def _generate_explainability_plots(cfg: dict, results: dict) -> None:
    """Generate SHAP, Grad-CAM, and correlation plots for trained models."""
    from src.evaluation.explainability import (
        plot_shap_summary,
        plot_shap_waterfall,
        plot_feature_correlation,
    )
    from src.utils import resolve_path
    import numpy as np

    plots_dir = resolve_path(cfg, "plots_out")
    plots_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["evaluation"]["plot_dpi"]

    # Load test data for plotting
    synthetic_dir = resolve_path(cfg, "synthetic_data")
    test_csv = synthetic_dir / "test.csv"
    if not test_csv.exists():
        logger.warning("No test CSV found for explainability plots")
        return

    import pandas as pd
    from src.data.loader import CHANNEL_NAMES
    from src.data.preprocessor import Preprocessor
    from src.features.extractor import FeatureExtractor

    test_df = pd.read_csv(test_csv)
    X_raw = test_df[CHANNEL_NAMES].values.astype(np.float32)
    y = test_df["label"].values.astype(np.int32)

    models_dir = resolve_path(cfg, "models_out")
    pipeline_path = models_dir / "preprocessing_pipeline.joblib"
    pipeline = Preprocessor.load_pipeline(pipeline_path) if pipeline_path.exists() else None

    extractor = FeatureExtractor(cfg)

    for model_name in ["random_forest", "xgboost"]:
        model_path = models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            continue

        try:
            from src.models.random_forest import RandomForestModel
            from src.models.xgboost_model import XGBoostModel
            model_cls = RandomForestModel if model_name == "random_forest" else XGBoostModel
            model = model_cls(cfg)
            model.load(model_path)

            if pipeline is not None:
                X_scaled = pipeline.transform(X_raw)
            else:
                X_scaled = X_raw

            from src.data.preprocessor import sliding_windows
            windows, labels = sliding_windows(
                X_scaled, y,
                cfg["preprocessing"]["window_size"],
                cfg["preprocessing"]["stride"],
            )
            X_feat = extractor.transform(windows)
            feature_names = extractor.get_feature_names()

            imp = model.get_feature_importance()

            plot_shap_summary(
                model, X_feat, feature_names,
                plots_dir / f"{model_name}_shap_summary.png", dpi=dpi,
            )
            plot_shap_waterfall(
                model, X_feat, labels, feature_names,
                plots_dir / f"{model_name}_shap_waterfall", dpi=dpi,
            )
            plot_feature_correlation(
                X_feat, feature_names, imp,
                plots_dir / f"{model_name}_feature_correlation.png", dpi=dpi,
            )

        except Exception as e:
            logger.warning("Explainability plots failed for %s: %s", model_name, e)


if __name__ == "__main__":
    main()
