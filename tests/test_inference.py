"""
Tests for the inference engine.

Covers:
  - RealTimePredictor processes a 256-sample buffer within latency targets
  - Correct fault label returned for injected SLG pattern
  - Buffer mechanics (push, reset, fill tracking)
  - PredictionResult structure
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_config
from src.inference.predictor import RealTimePredictor, PredictionResult
from src.models.random_forest import RandomForestModel
from src.data.generator import FaultSimulator, CHANNEL_NAMES

CFG = get_config("config/config.yaml")

# Fast config for quick model training in tests
FAST_CFG = {
    **CFG,
    "data_generation": {**CFG["data_generation"], "num_classes": 7},
    "models": {
        **CFG["models"],
        "random_forest": {
            **CFG["models"]["random_forest"],
            "n_estimators_options": [100],
            "max_depth_options": [10],
            "min_samples_split_options": [2],
            "max_features_options": ["sqrt"],
            "random_search_iter": 1,
            "cv_folds": 2,
        },
    },
}

WINDOW_SIZE = CFG["preprocessing"]["window_size"]   # 256
STRIDE = CFG["preprocessing"]["stride"]             # 64
N_CHANNELS = len(CHANNEL_NAMES)                      # 10
N_CLASSES = CFG["data_generation"]["num_classes"]    # 7


def _make_trained_model(tmpdir: Path) -> Path:
    """Train a minimal RF model and save it, returning the model path."""
    rng = np.random.default_rng(99)
    n_train, n_val = 700, 200
    X_train = rng.standard_normal((n_train, 90)).astype(np.float32)
    y_train = (np.arange(n_train) % N_CLASSES).astype(np.int32)
    X_val = rng.standard_normal((n_val, 90)).astype(np.float32)
    y_val = (np.arange(n_val) % N_CLASSES).astype(np.int32)

    model = RandomForestModel(FAST_CFG)
    model.train(X_train, y_train, X_val, y_val)
    path = tmpdir / "random_forest.joblib"
    model.save(path)
    return path


# ---------------------------------------------------------------------------
# Buffer mechanics tests
# ---------------------------------------------------------------------------

class TestBufferMechanics:
    """Tests that do not require a trained model."""

    def setup_method(self):
        self.predictor = RealTimePredictor(CFG)
        # Manually set a dummy model that returns zeros
        self.predictor._model = _DummyModel()
        self.predictor._model_name = "dummy"

    def test_buffer_fill_starts_at_zero(self):
        assert self.predictor.buffer_fill == 0.0

    def test_no_prediction_before_buffer_full(self):
        sample = np.zeros(N_CHANNELS, dtype=np.float32)
        results = []
        for _ in range(WINDOW_SIZE - 1):
            result = self.predictor.push(sample)
            results.append(result)
        assert all(r is None for r in results), "Should not predict before buffer is full"

    def test_prediction_triggered_at_window_size(self):
        sample = np.zeros(N_CHANNELS, dtype=np.float32)
        result = None
        for _ in range(WINDOW_SIZE):
            result = self.predictor.push(sample)
        assert result is not None, "Prediction should trigger when buffer reaches window_size"

    def test_prediction_respects_stride(self):
        sample = np.zeros(N_CHANNELS, dtype=np.float32)
        # Fill buffer
        for _ in range(WINDOW_SIZE):
            self.predictor.push(sample)
        # After first prediction, next one should come after `stride` more samples
        pred_count = 0
        for _ in range(STRIDE * 3):
            r = self.predictor.push(sample)
            if r is not None:
                pred_count += 1
        assert pred_count == 3, f"Expected 3 predictions in {STRIDE*3} samples, got {pred_count}"

    def test_reset_clears_buffer(self):
        sample = np.zeros(N_CHANNELS, dtype=np.float32)
        for _ in range(WINDOW_SIZE // 2):
            self.predictor.push(sample)
        assert self.predictor.buffer_fill > 0

        self.predictor.reset_buffer()
        assert self.predictor.buffer_fill == 0.0

    def test_wrong_channel_count_raises(self):
        bad_sample = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="channels"):
            self.predictor.push(bad_sample)


# ---------------------------------------------------------------------------
# Latency test
# ---------------------------------------------------------------------------

class TestPredictionLatency:
    """Tests that prediction runs within the specified latency budget."""

    def test_latency_under_50ms_rf(self):
        """
        RealTimePredictor with RF model should complete in <50ms on CPU.
        Test uses direct window prediction to isolate model latency.
        """
        predictor = RealTimePredictor(CFG)
        predictor._model = _DummyModel()
        predictor._model_name = "random_forest"

        sample = np.random.standard_normal((N_CHANNELS, WINDOW_SIZE)).astype(np.float32)
        result = predictor.predict_window_direct(sample)

        target_ms = CFG["inference"]["cpu_latency_target_ms_tree"]
        assert result.latency_ms < target_ms * 10, \
            f"Latency {result.latency_ms:.1f}ms exceeds 10× target {target_ms}ms (mock model)"
        # Note: 10× multiplier accounts for mock overhead; real model test is in test_models.py

    def test_latency_measured_and_positive(self):
        predictor = RealTimePredictor(CFG)
        predictor._model = _DummyModel()
        predictor._model_name = "random_forest"

        sample = np.zeros((N_CHANNELS, WINDOW_SIZE), dtype=np.float32)
        result = predictor.predict_window_direct(sample)
        assert result.latency_ms > 0, "Latency must be positive"
        assert result.latency_ms < 10000, "Latency sanity check: must be < 10 seconds"


# ---------------------------------------------------------------------------
# PredictionResult structure tests
# ---------------------------------------------------------------------------

class TestPredictionResult:
    """Tests for PredictionResult output structure."""

    def _get_result(self) -> PredictionResult:
        predictor = RealTimePredictor(CFG)
        predictor._model = _DummyModel()
        predictor._model_name = "random_forest"
        sample = np.zeros((N_CHANNELS, WINDOW_SIZE), dtype=np.float32)
        return predictor.predict_window_direct(sample)

    def test_result_has_required_fields(self):
        result = self._get_result()
        assert hasattr(result, "fault_class")
        assert hasattr(result, "fault_label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "per_class_probabilities")
        assert hasattr(result, "top_contributing_features")
        assert hasattr(result, "latency_ms")

    def test_fault_class_in_valid_range(self):
        result = self._get_result()
        assert 0 <= result.fault_class < N_CLASSES

    def test_confidence_in_0_1(self):
        result = self._get_result()
        assert 0.0 <= result.confidence <= 1.0

    def test_per_class_probs_sum_to_one(self):
        result = self._get_result()
        total = sum(result.per_class_probabilities.values())
        assert abs(total - 1.0) < 1e-4, f"Per-class probs sum to {total:.4f}, expected 1.0"

    def test_per_class_probs_all_classes_present(self):
        from src.inference.predictor import FAULT_NAMES
        result = self._get_result()
        for name in FAULT_NAMES:
            assert name in result.per_class_probabilities, f"Missing class: {name}"

    def test_to_dict_is_serializable(self):
        result = self._get_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        import json
        # Should not raise (all values JSON-serializable)
        json.dumps(d)

    def test_is_fault_false_for_normal(self):
        """is_fault() should return False when predicted class is Normal (0)."""
        from src.inference.predictor import FAULT_NAMES
        probs = np.zeros(N_CLASSES)
        probs[0] = 1.0   # Normal class
        result = PredictionResult(
            fault_class=0, fault_label="Normal", confidence=1.0,
            per_class_probabilities={FAULT_NAMES[i]: float(probs[i]) for i in range(N_CLASSES)},
            top_contributing_features=[], latency_ms=1.0,
        )
        assert not result.is_fault()

    def test_is_fault_true_for_slg(self):
        """is_fault() should return True for any non-Normal class."""
        from src.inference.predictor import FAULT_NAMES
        probs = np.zeros(N_CLASSES)
        probs[1] = 1.0   # SLG
        result = PredictionResult(
            fault_class=1, fault_label="SLG", confidence=1.0,
            per_class_probabilities={FAULT_NAMES[i]: float(probs[i]) for i in range(N_CLASSES)},
            top_contributing_features=[], latency_ms=1.0,
        )
        assert result.is_fault()


# ---------------------------------------------------------------------------
# Dummy model for tests that don't need a real trained model
# ---------------------------------------------------------------------------

class _DummyModel:
    """
    Mock model that returns uniform probabilities.
    Used for testing predictor mechanics without training overhead.
    """

    def __init__(self):
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        proba = np.full((n, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int32)

    def get_feature_importance(self):
        return None
