"""
Tests for model implementations.

Covers:
  - All models instantiate and train on mock data without error
  - predict_proba outputs valid probability distributions (sum to 1)
  - save/load cycle preserves predictions
  - BaseModel interface compliance
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_config
from src.models.base_model import BaseModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network import NeuralNetworkModel

CFG = get_config("config/config.yaml")

# Minimal config overrides for fast testing
FAST_CFG = {
    **CFG,
    "data_generation": {**CFG["data_generation"], "num_classes": 7},
    "models": {
        **CFG["models"],
        "random_forest": {
            **CFG["models"]["random_forest"],
            "n_estimators_options": [50],
            "max_depth_options": [5],
            "min_samples_split_options": [2],
            "max_features_options": ["sqrt"],
            "random_search_iter": 1,
            "cv_folds": 2,
        },
        "xgboost": {
            **CFG["models"]["xgboost"],
            "n_estimators_options": [50],
            "max_depth_options": [3],
            "learning_rate_options": [0.1],
            "subsample_options": [0.8],
            "colsample_bytree_options": [0.8],
            "early_stopping_rounds": 5,
            "random_search_iter": 1,
            "cv_folds": 2,
        },
        "neural_network": {
            **CFG["models"]["neural_network"],
            "epochs": 3,
            "batch_size": 64,
            "early_stopping_patience": 2,
            "conv_channels": [8, 16, 32],
            "kernel_sizes": [3, 3, 3],
            "fc_units": [16, 8],
            "dropout_rates": [0.1, 0.1],
            "mixed_precision": False,
        },
    },
}

N_TRAIN = 700
N_VAL = 200
N_CLASSES = 7
N_FEATURES = 90
WINDOW_SIZE = 256
N_CHANNELS = 10


def _mock_feature_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate (n, n_features) mock feature matrix with balanced labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    y = (np.arange(n) % N_CLASSES).astype(np.int32)
    return X, y


def _mock_window_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate (n, n_channels, window_size) mock window arrays for NN."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_CHANNELS, WINDOW_SIZE)).astype(np.float32)
    y = (np.arange(n) % N_CLASSES).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Base interface compliance
# ---------------------------------------------------------------------------

class TestBaseModelInterface:
    """Verify BaseModel is properly abstract."""

    def test_base_model_is_abstract(self):
        with pytest.raises(TypeError):
            BaseModel(CFG)   # Cannot instantiate abstract class

    def test_rf_inherits_base(self):
        assert issubclass(RandomForestModel, BaseModel)

    def test_xgb_inherits_base(self):
        assert issubclass(XGBoostModel, BaseModel)

    def test_nn_inherits_base(self):
        assert issubclass(NeuralNetworkModel, BaseModel)


# ---------------------------------------------------------------------------
# Random Forest tests
# ---------------------------------------------------------------------------

class TestRandomForestModel:
    """Tests for RandomForestModel."""

    def setup_method(self):
        self.X_train, self.y_train = _mock_feature_data(N_TRAIN, seed=1)
        self.X_val, self.y_val = _mock_feature_data(N_VAL, seed=2)
        self.model = RandomForestModel(FAST_CFG)

    def test_train_completes_without_error(self):
        metrics = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        assert "val_accuracy" in metrics
        assert 0.0 <= metrics["val_accuracy"] <= 1.0

    def test_predict_shape(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        preds = self.model.predict(self.X_val)
        assert preds.shape == (N_VAL,)
        assert set(preds).issubset(set(range(N_CLASSES)))

    def test_predict_proba_sums_to_one(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        proba = self.model.predict_proba(self.X_val)
        assert proba.shape == (N_VAL, N_CLASSES)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="predict_proba rows must sum to 1")

    def test_predict_proba_non_negative(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        proba = self.model.predict_proba(self.X_val)
        assert (proba >= 0).all(), "Probabilities must be non-negative"

    def test_save_load_preserves_predictions(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        preds_before = self.model.predict(self.X_val)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rf.joblib"
            self.model.save(path)

            loaded = RandomForestModel(FAST_CFG)
            loaded.load(path)
            preds_after = loaded.predict(self.X_val)

        np.testing.assert_array_equal(preds_before, preds_after,
                                      err_msg="Predictions changed after save/load")

    def test_feature_importance_shape(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        imp = self.model.get_feature_importance()
        assert imp is not None
        assert imp.shape == (N_FEATURES,)
        assert (imp >= 0).all(), "Importances must be non-negative"

    def test_unfitted_predict_raises(self):
        model = RandomForestModel(FAST_CFG)
        with pytest.raises(RuntimeError):
            model.predict(self.X_val)


# ---------------------------------------------------------------------------
# XGBoost tests
# ---------------------------------------------------------------------------

class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def setup_method(self):
        self.X_train, self.y_train = _mock_feature_data(N_TRAIN, seed=3)
        self.X_val, self.y_val = _mock_feature_data(N_VAL, seed=4)
        self.model = XGBoostModel(FAST_CFG)

    def test_train_completes_without_error(self):
        metrics = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        assert "val_accuracy" in metrics

    def test_predict_proba_sums_to_one(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        proba = self.model.predict_proba(self.X_val)
        assert proba.shape == (N_VAL, N_CLASSES)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_save_load_preserves_predictions(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        preds_before = self.model.predict(self.X_val)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xgb.joblib"
            self.model.save(path)
            loaded = XGBoostModel(FAST_CFG)
            loaded.load(path)
            preds_after = loaded.predict(self.X_val)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_feature_importance_shape(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        imp = self.model.get_feature_importance()
        assert imp is not None
        assert imp.shape == (N_FEATURES,)


# ---------------------------------------------------------------------------
# Neural Network tests
# ---------------------------------------------------------------------------

class TestNeuralNetworkModel:
    """Tests for NeuralNetworkModel (Temporal CNN)."""

    def setup_method(self):
        self.X_train, self.y_train = _mock_window_data(N_TRAIN, seed=5)
        self.X_val, self.y_val = _mock_window_data(N_VAL, seed=6)
        self.model = NeuralNetworkModel(FAST_CFG)

    def test_train_completes_without_error(self):
        metrics = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        assert "best_val_f1" in metrics
        assert 0.0 <= metrics["best_val_f1"] <= 1.0

    def test_predict_shape(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        preds = self.model.predict(self.X_val)
        assert preds.shape == (N_VAL,)

    def test_predict_proba_sums_to_one(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        proba = self.model.predict_proba(self.X_val)
        assert proba.shape == (N_VAL, N_CLASSES)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_predict_proba_non_negative(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        proba = self.model.predict_proba(self.X_val)
        assert (proba >= 0).all()

    def test_save_load_preserves_predictions(self):
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        preds_before = self.model.predict(self.X_val)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nn.pt"
            self.model.save(path)
            loaded = NeuralNetworkModel(FAST_CFG)
            loaded.load(path)
            preds_after = loaded.predict(self.X_val)

        np.testing.assert_array_equal(preds_before, preds_after,
                                      err_msg="NN predictions changed after save/load")

    def test_feature_importance_returns_none(self):
        """NN does not have simple feature importances (uses Grad-CAM instead)."""
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        imp = self.model.get_feature_importance()
        assert imp is None
