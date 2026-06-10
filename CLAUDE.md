# CLAUDE.md — AI Fault Prediction in Power Systems

This file gives Claude Code sessions full context to resume work without re-explanation.

## Project Purpose

End-to-end ML pipeline for predictive fault detection in electrical power grids.
Targets integration with IEC 61850-compliant protection systems and SCADA platforms.
Fault detection replaces threshold-based relay logic with learned patterns.

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/data/generator.py` | Synthetic power system fault dataset generation (7 classes, 1 kHz) |
| `src/data/loader.py` | CSV loading, schema validation, train/val/test splitting |
| `src/data/preprocessor.py` | Missing value handling, outlier clipping, RobustScaler, sliding windows |
| `src/features/extractor.py` | Time-domain, frequency-domain, power-system-specific feature extraction |
| `src/features/selector.py` | Feature importance ranking, correlation-based selection |
| `src/models/base_model.py` | Abstract BaseModel with required interface (train/predict/save/load/importance) |
| `src/models/random_forest.py` | RF with RandomizedSearchCV + isotonic calibration |
| `src/models/xgboost_model.py` | XGBoost with early stopping + SHAP values |
| `src/models/neural_network.py` | Temporal CNN in PyTorch with AdamW + CosineAnnealing |
| `src/training/trainer.py` | Stratified 5-fold CV unified training loop |
| `src/training/cross_validator.py` | Cross-validation utilities |
| `src/evaluation/metrics.py` | All metrics: accuracy, FDR, FAR, MTTD, ECE, calibration |
| `src/evaluation/explainability.py` | SHAP plots, Grad-CAM for CNN, feature correlation heatmap |
| `src/inference/predictor.py` | RealTimePredictor: buffer → window → predict → SHAP |
| `src/dashboard/app.py` | Streamlit monitoring dashboard |
| `run_pipeline.py` | Single entry point: `python run_pipeline.py --config config/config.yaml` |

## Fault Classes

| ID | Name | Key Signature |
|----|------|--------------|
| 0 | Normal | All signals nominal, Gaussian noise |
| 1 | Single Line-to-Ground (SLG) | One phase V drops 30-60%, I spikes 2-4×, neutral current rises |
| 2 | Line-to-Line (LL) | Two phases V dip 20-40%, current imbalance, ±0.1 Hz freq |
| 3 | Three-Phase (3PH) | All V collapse >80%, all I spike 5-10×, ±0.3 Hz freq |
| 4 | Overload | Gradual I rise over 2-10s, PF degradation |
| 5 | Voltage Sag | V drops 10-30% for 0.1-2s |
| 6 | Harmonic Distortion | THD >8%, 3rd/5th/7th harmonics prominent |

## Sensor Channels (9 total)

Va, Vb, Vc (phase voltages), Ia, Ib, Ic (phase currents), Frequency, Power Factor, THD_V, Neutral Current

Note: generator outputs 10 columns (Va,Vb,Vc,Ia,Ib,Ic,Freq,PF,THD_V,In) + label.

## Key Design Decisions

- **Symmetrical components** (Fortescue transform) are the most discriminative features for fault type classification — they mirror conventional relay protection mathematics.
- **ROCOF** (Rate of Change of Frequency) is sensitive to inertia events preceding major faults.
- **Isotonic calibration** on RF ensures `predict_proba` outputs are trustworthy for alarm thresholds.
- **Temporal CNN** preferred over MLP because it captures temporal dynamics within each window.
- **RobustScaler** chosen over StandardScaler because fault spikes should not distort normalization.
- **MTTD** (Mean Time to Detection) corresponds to relay operating time — a safety-critical metric.

## Build Order

1. Phase 1: `src/data/generator.py` → validate with `python -m src.data.generator --validate`
2. Phase 2: `src/data/preprocessor.py`, `src/features/extractor.py` → validate with `python -m src.features.extractor --test`
3. Phase 3: `src/models/` → validate with `pytest tests/test_models.py -v`
4. Phase 4: `src/training/`, `src/evaluation/` → validate with `python -m src.training.trainer --dry-run`
5. Phase 5: `src/inference/`, `src/dashboard/` → validate with `pytest tests/test_inference.py -v`
6. Phase 6: `tests/` full suite → `pytest tests/ -v`
7. End-to-end: `python run_pipeline.py --config config/config.yaml`

## Verification Gates

| Gate | Command | Criterion |
|------|---------|-----------|
| Phase 1 | `python -m src.data.generator --validate` | 7 classes, no NaN, ≥50k/class |
| Phase 2 | `python -m src.features.extractor --test` | 80-120 features, no NaN/Inf |
| Phase 3 | `pytest tests/test_models.py -v` | 100% pass |
| Phase 4 | `python -m src.training.trainer --dry-run` | FDR>90%, FAR<5% |
| Phase 5 | `pytest tests/test_inference.py -v` | All pass, latency <50ms |
| Phase 6 | `pytest tests/ -v` | Full suite green |

## Domain Context

This system is analogous to a numerical protection relay implementing:
- Overcurrent protection (Class 1/2/3/4 detection)
- Undervoltage protection (Class 3/5 detection)  
- Power quality monitoring (Class 6 detection)
- The ML model learns the same symmetrical-component mathematics that IEC relays use,
  making predictions interpretable to protection engineers.
