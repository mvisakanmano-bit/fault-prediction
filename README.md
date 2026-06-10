# AI-Based Fault Prediction in Power Systems

End-to-end machine learning pipeline for predictive fault detection in electrical power grids.
Integrable with IEC 61850-compliant SCADA and protection relay systems.

## Project Purpose

Traditional relay protection uses fixed thresholds (overcurrent, undervoltage) that cannot distinguish
fault types or predict incipient faults before they cause outages. This pipeline trains three ML models
on multivariate power system sensor data to classify seven fault conditions with >95% detection rate
and <2% false alarm rate — matching the performance requirements of numerical protection relays.

## Setup

```bash
# 1. Clone or copy repository
cd fault_prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for the right command
```

## Dataset

No external dataset required. The pipeline generates a realistic synthetic power system dataset:

| Property | Value |
|----------|-------|
| Sample rate | 1 kHz (1000 samples/second) |
| Channels | 10 (Va, Vb, Vc, Ia, Ib, Ic, Freq, PF, THD_V, In) |
| Classes | 7 (Normal + 6 fault types) |
| Samples per class | 50,000 minimum |
| Total dataset | ~350,000 samples |
| Split | 70% train / 15% val / 15% test (stratified) |

### Fault Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Normal | All signals nominal with Gaussian noise |
| 1 | SLG | Single Line-to-Ground: one phase V↓30–60%, I↑2–4× |
| 2 | LL | Line-to-Line: two phases V↓20–40%, current imbalance |
| 3 | 3PH | Three-Phase: all V collapse >80%, all I spike 5–10× |
| 4 | Overload | Gradual current rise over 2–10s, PF degradation |
| 5 | Voltage Sag | V drops 10–30% for 0.1–2 seconds |
| 6 | Harmonic | THD >8%, 3rd/5th/7th harmonics prominent |

## Model Architectures

| Model | Framework | Key Features |
|-------|-----------|-------------|
| Random Forest | scikit-learn | 200–1000 trees, isotonic probability calibration |
| XGBoost | xgboost | Early stopping, SHAP values, class-weight balancing |
| Temporal CNN | PyTorch | 3× Conv1D layers, AdamW + CosineAnnealing, mixed precision |

## Training Commands

```bash
# Generate synthetic dataset
python -m src.data.generator --validate

# Extract features
python -m src.features.extractor --test

# Train all models (full pipeline)
python run_pipeline.py --config config/config.yaml

# Train a single model
python run_pipeline.py --config config/config.yaml --model random_forest

# Dry run (1 CV fold, quick validation)
python -m src.training.trainer --dry-run
```

## Evaluation Results

*Populated after first full training run*

| Metric | Random Forest | XGBoost | Neural Network |
|--------|--------------|---------|----------------|
| Accuracy | — | — | — |
| F1 Macro | — | — | — |
| MCC | — | — | — |
| FDR (avg) | — | — | — |
| FAR (avg) | — | — | — |
| MTTD (ms) | — | — | — |

## Dashboard

```bash
# Launch monitoring dashboard
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501)

Features:
- Live signal plots with rolling 5-second window
- Per-class fault probability gauges (green/yellow/red)
- Alert log with timestamps, fault type, confidence, contributing features
- Model switcher (RF / XGBoost / NN)
- Historical fault trend chart
- Batch CSV analysis mode

## Testing

```bash
# Run full test suite
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Project Structure

```
fault_prediction/
├── config/config.yaml          # All hyperparameters and thresholds
├── src/
│   ├── data/                   # Generation, loading, preprocessing
│   ├── features/               # Feature extraction and selection
│   ├── models/                 # RF, XGBoost, Neural Network
│   ├── training/               # Training loop and cross-validation
│   ├── evaluation/             # Metrics and explainability
│   ├── inference/              # Real-time prediction interface
│   └── dashboard/              # Streamlit monitoring app
├── tests/                      # pytest test suite
├── outputs/
│   ├── models/                 # Serialized model files
│   ├── reports/                # HTML evaluation reports
│   └── plots/                  # All figures (300 DPI)
└── run_pipeline.py             # Single entry point
```

## Domain Context

The symmetrical components (Fortescue transform) features extracted by this pipeline directly
mirror the mathematics used in IEC numerical protection relays. Positive/negative/zero sequence
voltages are the primary discriminators for fault type classification in conventional relay logic.
The MTTD metric corresponds to relay operating time — a safety-critical parameter in protection
coordination studies.
