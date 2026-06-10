"""
Real-time fault prediction inference engine.

RealTimePredictor maintains a rolling sample buffer, triggers prediction
when a full window is accumulated, and returns structured results including
fault class, confidence, and top contributing SHAP features.

Latency targets:
  - RF / XGBoost: <50ms per prediction on CPU
  - Neural Network: <100ms per prediction on CPU
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FAULT_NAMES = ["Normal", "SLG", "LL", "3PH", "Overload", "VoltageSag", "Harmonic"]
CHANNEL_NAMES = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Freq", "PF", "THD_V", "In"]


class PredictionResult:
    """Structured output from a single window prediction."""

    def __init__(
        self,
        fault_class: int,
        fault_label: str,
        confidence: float,
        per_class_probabilities: Dict[str, float],
        top_contributing_features: List[Tuple[str, float]],
        latency_ms: float,
    ):
        self.fault_class = fault_class
        self.fault_label = fault_label
        self.confidence = confidence
        self.per_class_probabilities = per_class_probabilities
        self.top_contributing_features = top_contributing_features
        self.latency_ms = latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_class": self.fault_class,
            "fault_label": self.fault_label,
            "confidence": self.confidence,
            "per_class_probabilities": self.per_class_probabilities,
            "top_contributing_features": self.top_contributing_features,
            "latency_ms": self.latency_ms,
        }

    def is_fault(self) -> bool:
        return self.fault_class != 0

    def __repr__(self) -> str:
        return (
            f"PredictionResult(class={self.fault_class}, label={self.fault_label!r}, "
            f"conf={self.confidence:.3f}, latency={self.latency_ms:.1f}ms)"
        )


class RealTimePredictor:
    """
    Streaming fault predictor with rolling window buffer.

    Usage:
        predictor = RealTimePredictor.from_config(cfg)
        predictor.load_model("random_forest", models_dir)

        # Feed samples one at a time or in batches
        for sample in sensor_stream:
            result = predictor.push(sample)   # None until buffer is full
            if result is not None:
                print(result)
    """

    def __init__(
        self,
        cfg: dict,
        model_name: Optional[str] = None,
    ):
        self.cfg = cfg
        self.window_size = cfg["preprocessing"]["window_size"]
        self.stride = cfg["preprocessing"]["stride"]
        self.n_channels = len(CHANNEL_NAMES)
        self.top_n_features = cfg["inference"]["top_features_n"]

        # Rolling buffer: holds the last window_size samples (each sample is a row of n_channels values)
        self._buffer: deque = deque(maxlen=self.window_size)
        self._samples_since_last_pred: int = 0

        self._model: Optional[Any] = None
        self._model_name: Optional[str] = None
        self._feature_extractor: Optional[Any] = None
        self._feature_names: Optional[List[str]] = None
        self._preprocessing_pipeline: Optional[Any] = None
        self._shap_explainer: Optional[Any] = None

        if model_name:
            self._model_name = model_name

    @classmethod
    def from_config(cls, cfg: dict) -> "RealTimePredictor":
        """Create predictor using default model from config."""
        return cls(cfg, model_name=cfg["inference"]["default_model"])

    def load_model(self, model_name: str, models_dir: Path) -> None:
        """
        Load a trained model by name from the models directory.

        Also attempts to load the preprocessing pipeline and feature extractor.
        """
        from src.models.random_forest import RandomForestModel
        from src.models.xgboost_model import XGBoostModel
        from src.models.neural_network import NeuralNetworkModel

        model_registry = {
            "random_forest": (RandomForestModel, models_dir / "random_forest.joblib"),
            "xgboost": (XGBoostModel, models_dir / "xgboost.joblib"),
            "neural_network": (NeuralNetworkModel, models_dir / "neural_network.pt"),
        }

        if model_name not in model_registry:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_registry)}")

        model_cls, model_path = model_registry[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = model_cls(self.cfg)
        model.load(model_path)
        self._model = model
        self._model_name = model_name
        logger.info("Loaded model: %s from %s", model_name, model_path)

        # Load preprocessing pipeline
        pipeline_path = models_dir / "preprocessing_pipeline.joblib"
        if pipeline_path.exists():
            from src.data.preprocessor import Preprocessor
            self._preprocessing_pipeline = Preprocessor.load_pipeline(pipeline_path)
            logger.info("Preprocessing pipeline loaded")

        # Initialize feature extractor
        from src.features.extractor import FeatureExtractor
        self._feature_extractor = FeatureExtractor(self.cfg)
        self._feature_names = self._feature_extractor.get_feature_names()

        # Build SHAP explainer if available
        self._build_shap_explainer()

    def _build_shap_explainer(self) -> None:
        """Attempt to build a SHAP explainer for the loaded model."""
        if self._model_name == "neural_network":
            return  # Grad-CAM used for NN instead

        try:
            import shap
            underlying = getattr(self._model, "_base_model", None) or getattr(self._model, "model", None)
            if underlying is not None:
                self._shap_explainer = shap.TreeExplainer(underlying)
                logger.info("SHAP explainer ready")
        except Exception as e:
            logger.debug("SHAP explainer unavailable: %s", e)

    def push(self, sample: np.ndarray) -> Optional[PredictionResult]:
        """
        Add one sample (n_channels,) to the buffer and trigger prediction
        when stride samples have accumulated since the last prediction.

        Args:
            sample: (n_channels,) sensor readings for one timestep.

        Returns:
            PredictionResult if a prediction was triggered, else None.
        """
        if sample.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {sample.shape[0]}")

        self._buffer.append(sample.astype(np.float32))
        self._samples_since_last_pred += 1

        # Trigger prediction when buffer is full and stride has elapsed
        if len(self._buffer) >= self.window_size and self._samples_since_last_pred >= self.stride:
            self._samples_since_last_pred = 0
            window = np.array(self._buffer, dtype=np.float32)  # (window_size, n_channels)
            return self._predict_window(window)

        return None

    def push_batch(self, samples: np.ndarray) -> List[Optional[PredictionResult]]:
        """
        Feed multiple samples at once.

        Args:
            samples: (n_samples, n_channels) array.

        Returns:
            List of PredictionResult or None for each sample.
        """
        return [self.push(samples[i]) for i in range(len(samples))]

    def predict_window_direct(self, window: np.ndarray) -> PredictionResult:
        """
        Run prediction directly on a pre-assembled window.

        Args:
            window: (window_size, n_channels) array OR (n_channels, window_size) channel-first.

        Returns:
            PredictionResult
        """
        if window.shape[0] == self.n_channels:
            # Channel-first: transpose to (window_size, n_channels)
            window = window.T
        return self._predict_window(window)

    def _predict_window(self, window: np.ndarray) -> PredictionResult:
        """
        Internal: apply pipeline → extract features → predict.

        window: (window_size, n_channels) row-major.
        """
        t0 = time.perf_counter()

        # Preprocess
        if self._preprocessing_pipeline is not None:
            try:
                window_scaled = self._preprocessing_pipeline.transform(window)
            except Exception:
                window_scaled = window
        else:
            window_scaled = window

        # Feature extraction for tree models; raw windows for NN
        if self._model_name == "neural_network":
            # NN expects (1, n_channels, window_size)
            X_input = window_scaled.T[np.newaxis]   # (1, n_ch, win)
        else:
            # Feature-based models
            win_ch_first = window_scaled.T           # (n_ch, win)
            if self._feature_extractor is not None:
                features = self._feature_extractor.extract_window(win_ch_first)
            else:
                features = win_ch_first.flatten()
            X_input = features[np.newaxis]           # (1, n_features)

        # Predict
        proba = self._model.predict_proba(X_input)[0]  # (n_classes,)
        fault_class = int(np.argmax(proba))
        confidence = float(proba[fault_class])

        per_class_probs = {FAULT_NAMES[i]: float(proba[i]) for i in range(len(FAULT_NAMES))}

        # SHAP top features
        top_features = self._get_top_features(X_input if self._model_name != "neural_network" else None)

        latency_ms = (time.perf_counter() - t0) * 1000

        return PredictionResult(
            fault_class=fault_class,
            fault_label=FAULT_NAMES[fault_class],
            confidence=confidence,
            per_class_probabilities=per_class_probs,
            top_contributing_features=top_features,
            latency_ms=latency_ms,
        )

    def _get_top_features(self, X: Optional[np.ndarray]) -> List[Tuple[str, float]]:
        """Return top contributing features from SHAP or feature importances."""
        if X is None or self._feature_names is None:
            return []

        # Try SHAP explanation
        if self._shap_explainer is not None:
            try:
                shap_vals = self._shap_explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    # Multi-class: use max absolute contribution across classes
                    sv = np.max(np.abs(np.stack(shap_vals, axis=0)), axis=0)[0]
                else:
                    sv = np.abs(shap_vals[0])

                top_idx = np.argsort(sv)[::-1][:self.top_n_features]
                return [(self._feature_names[i], float(sv[i])) for i in top_idx]
            except Exception:
                pass

        # Fall back to raw feature values if SHAP unavailable
        if X is not None and len(self._feature_names) == X.shape[1]:
            vals = np.abs(X[0])
            top_idx = np.argsort(vals)[::-1][:self.top_n_features]
            return [(self._feature_names[i], float(vals[i])) for i in top_idx]

        return []

    def reset_buffer(self) -> None:
        """Clear the sample buffer (e.g., after a configuration change)."""
        self._buffer.clear()
        self._samples_since_last_pred = 0

    @property
    def buffer_fill(self) -> float:
        """Buffer fullness as a fraction [0, 1]."""
        return len(self._buffer) / self.window_size
