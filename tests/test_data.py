"""
Tests for data generation and preprocessing.

Covers:
  - Generator class distribution and NaN validation
  - Preprocessor missing value handling
  - Sliding window output shape
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_config
from src.data.generator import DatasetGenerator, FaultSimulator, FAULT_CLASSES, CHANNEL_NAMES
from src.data.preprocessor import (
    MissingValueHandler,
    IQRClipper,
    build_preprocessing_pipeline,
    sliding_windows,
    Preprocessor,
)

CFG = get_config("config/config.yaml")
# Use small samples for test speed
SMALL_CFG = {**CFG}
SMALL_CFG["data_generation"] = {**CFG["data_generation"], "samples_per_class": 2000}


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestDataGenerator:
    """Tests for FaultSimulator and DatasetGenerator."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)

    def test_all_fault_classes_present(self):
        """Generator must produce all 7 fault classes."""
        gen = DatasetGenerator(SMALL_CFG)
        df = gen.generate()
        classes_found = set(df["label"].unique())
        assert classes_found == set(FAULT_CLASSES.keys()), \
            f"Missing classes: {set(FAULT_CLASSES.keys()) - classes_found}"

    def test_no_nan_in_sensor_columns(self):
        """Generated sensor data must be free of NaN values."""
        gen = DatasetGenerator(SMALL_CFG)
        df = gen.generate()
        nan_counts = df[CHANNEL_NAMES].isnull().sum()
        assert nan_counts.sum() == 0, f"NaN found in: {nan_counts[nan_counts > 0].to_dict()}"

    def test_minimum_samples_per_class(self):
        """Each class must have at least samples_per_class rows."""
        gen = DatasetGenerator(SMALL_CFG)
        df = gen.generate()
        min_required = SMALL_CFG["data_generation"]["samples_per_class"]
        class_counts = df["label"].value_counts()
        for cls_id in FAULT_CLASSES:
            count = class_counts.get(cls_id, 0)
            assert count >= min_required, \
                f"Class {cls_id} has {count} samples, need {min_required}"

    def test_output_shape(self):
        """Output DataFrame must have correct number of columns."""
        gen = DatasetGenerator(SMALL_CFG)
        df = gen.generate()
        # Va, Vb, Vc, Ia, Ib, Ic, Freq, PF, THD_V, In + label + fault_name
        assert "label" in df.columns
        for ch in CHANNEL_NAMES:
            assert ch in df.columns, f"Missing channel: {ch}"

    def test_normal_class_voltage_range(self):
        """Normal class voltages must stay within defined bounds (with tolerance for noise)."""
        sim = FaultSimulator(CFG, self.rng)
        data = sim.normal(1000)
        gen_cfg = CFG["data_generation"]
        # Allow 3-sigma slack for normal noise
        for col_idx in [0, 1, 2]:  # Va, Vb, Vc
            rms_val = float(np.sqrt(np.mean(data[:, col_idx] ** 2)))
            assert rms_val > 100, f"Voltage RMS {rms_val:.1f}V seems too low for nominal {gen_cfg['voltage_nominal']}V"

    def test_slg_fault_voltage_drop(self):
        """SLG fault must produce a phase A voltage drop."""
        sim = FaultSimulator(CFG, self.rng)
        n = CFG["data_generation"]["sample_rate"] * 2  # 2 seconds
        data = sim.single_line_to_ground(n)
        pre = CFG["data_generation"]["pre_fault_ms"] * CFG["data_generation"]["sample_rate"] // 1000
        # Phase A voltage RMS during fault should be lower than during pre-fault
        rms_pre = float(np.sqrt(np.mean(data[:pre, 0] ** 2)))
        fault_start = pre + 50
        fault_end = pre + 500
        rms_fault = float(np.sqrt(np.mean(data[fault_start:fault_end, 0] ** 2)))
        assert rms_fault < rms_pre * 0.9, \
            f"SLG fault did not reduce Va RMS: pre={rms_pre:.2f}, fault={rms_fault:.2f}"

    def test_stratified_split_proportions(self):
        """Train/val/test split must maintain specified proportions (±2%)."""
        gen = DatasetGenerator(SMALL_CFG)
        df = gen.generate()
        train_df, val_df, test_df = gen.split(df)
        total = len(df)
        train_ratio = len(train_df) / total
        val_ratio = len(val_df) / total
        expected_train = SMALL_CFG["data_generation"]["train_ratio"]
        expected_val = SMALL_CFG["data_generation"]["val_ratio"]
        assert abs(train_ratio - expected_train) < 0.02, f"Train ratio {train_ratio:.3f} ≠ {expected_train}"
        assert abs(val_ratio - expected_val) < 0.02, f"Val ratio {val_ratio:.3f} ≠ {expected_val}"


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    """Tests for preprocessing pipeline components."""

    def test_missing_value_forward_fill(self):
        """Forward-fill must handle gaps up to max_fill samples."""
        handler = MissingValueHandler(max_fill=5)
        handler.fit(None)
        X = np.ones((20, 3), dtype=float)
        X[5:9, 1] = np.nan   # 4 consecutive NaN (within fill limit)
        result = handler.transform(X)
        assert not np.isnan(result).any(), "Forward-fill should have handled 4-sample gap"

    def test_missing_value_long_gap_dropped(self):
        """Rows in segments with >max_fill consecutive NaN must be dropped."""
        handler = MissingValueHandler(max_fill=3)
        handler.fit(None)
        X = np.ones((20, 3), dtype=float)
        X[5:15, 0] = np.nan  # 10 consecutive NaN > max_fill=3
        result = handler.transform(X)
        # Some rows should be dropped
        assert result.shape[0] < 20, "Long NaN gap should have caused row drops"

    def test_iqr_clipper_removes_outliers(self):
        """IQR clipper must constrain outliers to [Q1-3IQR, Q3+3IQR]."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((1000, 5))
        X_test = rng.standard_normal((100, 5))
        X_test[0, :] = 1000   # Extreme outlier

        clipper = IQRClipper(k=3.0)
        clipper.fit(X_train)
        X_clipped = clipper.transform(X_test)

        assert X_clipped[0, :].max() < 10, "Extreme outlier was not clipped"
        assert not np.isinf(X_clipped).any()

    def test_pipeline_fit_transform(self):
        """Full preprocessing pipeline must produce finite output."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 10)).astype(np.float32)
        pipeline = build_preprocessing_pipeline(CFG)
        X_out = pipeline.fit_transform(X)
        assert np.isfinite(X_out).all(), "Pipeline output contains non-finite values"
        assert X_out.shape == X.shape, "Pipeline changed array shape unexpectedly"

    def test_sliding_windows_shape(self):
        """Sliding windows must produce correct (n_windows, n_ch, win_size) shape."""
        n_samples = 1000
        n_channels = 10
        window_size = 256
        stride = 64
        X = np.random.rand(n_samples, n_channels).astype(np.float32)
        y = np.zeros(n_samples, dtype=np.int32)

        windows, labels = sliding_windows(X, y, window_size, stride)

        expected_n_windows = (n_samples - window_size) // stride + 1
        assert windows.shape == (expected_n_windows, n_channels, window_size), \
            f"Expected shape ({expected_n_windows}, {n_channels}, {window_size}), got {windows.shape}"
        assert labels.shape == (expected_n_windows,)

    def test_window_label_majority_vote(self):
        """Window label should be determined by majority class in the window span."""
        n = 512
        window_size = 256
        stride = 256
        X = np.zeros((n, 4), dtype=np.float32)
        y = np.zeros(n, dtype=np.int32)
        y[200:256] = 1   # Fault in last part of first window

        windows, labels = sliding_windows(X, y, window_size, stride)
        # First window has 200 normal + 56 fault → normal (class 0) wins
        assert labels[0] == 0, "Majority-vote label should be 0 (Normal)"

    def test_preprocessor_transform_consistency(self):
        """Pipeline fitted on train must transform val without error."""
        rng = np.random.default_rng(99)
        X_train = rng.standard_normal((500, 10)).astype(np.float32)
        X_val = rng.standard_normal((100, 10)).astype(np.float32)
        y_train = np.zeros(500, dtype=np.int32)
        y_val = np.zeros(100, dtype=np.int32)

        prep = Preprocessor(CFG)
        windows_train, labels_train, _ = prep.fit_transform(X_train, y_train)
        windows_val, labels_val = prep.transform(X_val, y_val)

        assert windows_train.ndim == 3
        assert windows_val.ndim == 3
        assert windows_train.shape[1] == windows_val.shape[1], "Channel count mismatch"
