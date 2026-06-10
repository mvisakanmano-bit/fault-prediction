"""
Tests for feature extraction.

Covers:
  - THD computation against manual calculation
  - Symmetrical components correctness for known unbalanced inputs
  - Feature vector shape consistency
  - No NaN/Inf in feature output
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_config
from src.features.extractor import FeatureExtractor, _symmetrical_components, _phasor_from_window

CFG = get_config("config/config.yaml")


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""

    def setup_method(self):
        self.extractor = FeatureExtractor(CFG)
        self.window_size = CFG["preprocessing"]["window_size"]   # 256
        self.n_ch = 10
        self.fs = CFG["data_generation"]["sample_rate"]
        self.f0 = CFG["features"]["fundamental_freq"]

    def _make_window(self, override: dict = None) -> np.ndarray:
        """Create a synthetic 3-phase window. override allows channel-specific data."""
        t = np.arange(self.window_size) / self.fs
        v_nom = CFG["data_generation"]["voltage_nominal"]
        i_nom = 50.0
        win = np.zeros((self.n_ch, self.window_size), dtype=np.float32)
        # Va, Vb, Vc
        win[0] = (v_nom * np.sin(2 * np.pi * self.f0 * t)).astype(np.float32)
        win[1] = (v_nom * np.sin(2 * np.pi * self.f0 * t - 2.094)).astype(np.float32)
        win[2] = (v_nom * np.sin(2 * np.pi * self.f0 * t + 2.094)).astype(np.float32)
        # Ia, Ib, Ic
        win[3] = (i_nom * np.sin(2 * np.pi * self.f0 * t - 0.524)).astype(np.float32)
        win[4] = (i_nom * np.sin(2 * np.pi * self.f0 * t - 0.524 - 2.094)).astype(np.float32)
        win[5] = (i_nom * np.sin(2 * np.pi * self.f0 * t - 0.524 + 2.094)).astype(np.float32)
        # Freq, PF, THD, In
        win[6] = np.full(self.window_size, 50.0, dtype=np.float32)
        win[7] = np.full(self.window_size, 0.95, dtype=np.float32)
        win[8] = np.full(self.window_size, 2.0, dtype=np.float32)
        win[9] = np.full(self.window_size, 0.5, dtype=np.float32)
        if override:
            for idx, arr in override.items():
                win[idx] = arr
        return win

    def test_feature_vector_shape_consistent(self):
        """All windows must produce the same feature vector length."""
        rng = np.random.default_rng(0)
        shapes = set()
        for _ in range(5):
            dummy = rng.standard_normal((self.n_ch, self.window_size)).astype(np.float32)
            feats = self.extractor.extract_window(dummy)
            shapes.add(len(feats))
        assert len(shapes) == 1, f"Inconsistent feature vector sizes: {shapes}"

    def test_feature_count_in_range(self):
        """Feature count must be within the configured range."""
        win = self._make_window()
        feats = self.extractor.extract_window(win)
        n = len(feats)
        min_f = CFG["features"]["target_feature_count_min"]
        max_f = CFG["features"]["target_feature_count_max"]
        assert min_f <= n <= max_f, f"Feature count {n} outside [{min_f}, {max_f}]"

    def test_no_nan_in_features(self):
        """Normal signal window must produce no NaN features."""
        win = self._make_window()
        feats = self.extractor.extract_window(win)
        assert not np.isnan(feats).any(), "NaN detected in feature vector from normal signal"

    def test_no_inf_in_features(self):
        """Normal signal window must produce no Inf features."""
        win = self._make_window()
        feats = self.extractor.extract_window(win)
        assert not np.isinf(feats).any(), "Inf detected in feature vector from normal signal"

    def test_thd_fft_matches_manual(self):
        """
        THD computed from FFT should approximately match manual calculation.

        For signal = sin(2π*50t) + 0.10*sin(2π*150t) + 0.05*sin(2π*250t),
        manual THD = sqrt(0.10² + 0.05²) / 1.0 * 100 ≈ 11.18%
        """
        t = np.arange(self.window_size) / self.fs
        h3_amp = 0.10
        h5_amp = 0.05
        signal = np.sin(2 * np.pi * self.f0 * t) + h3_amp * np.sin(2 * np.pi * 150 * t) + h5_amp * np.sin(2 * np.pi * 250 * t)
        manual_thd = np.sqrt(h3_amp ** 2 + h5_amp ** 2) * 100  # ≈ 11.18%

        feats = self.extractor._fft_features(signal.astype(np.float32), "test")
        fft_thd = feats["test_thd_fft"]

        # Allow ±5 percentage points tolerance (FFT window leakage)
        assert abs(fft_thd - manual_thd) < 5.0, \
            f"THD mismatch: FFT={fft_thd:.2f}%, manual={manual_thd:.2f}%"

    def test_symmetrical_components_balanced(self):
        """
        For balanced positive-sequence 3-phase input, negative and zero sequence
        magnitudes should be near zero; positive sequence ≈ 1/3 of applied magnitude.

        Standard Fortescue: Va=1∠0°, Vb=1∠-120°=α², Vc=1∠+120°=α
        where α = e^{j2π/3}.
        V1 = (Va + α·Vb + α²·Vc)/3 = (1 + α·α² + α²·α)/3 = (1+α³+α³)/3 = 1 ✓
        """
        alpha = np.exp(1j * 2 * np.pi / 3)
        # Balanced positive sequence: Vb = α² = 1∠-120°, Vc = α = 1∠+120°
        va = complex(1, 0)
        vb = alpha ** 2          # 1∠-120°
        vc = alpha               # 1∠+120°

        v0, v1, v2 = _symmetrical_components(va, vb, vc)

        # Positive sequence magnitude should be 1 (sum of three aligned unit phasors / 3 = 1)
        assert abs(v1) > 0.9, f"Positive sequence {abs(v1):.6f} should be ~1 for balanced input"
        assert abs(v2) < 0.05, f"Negative sequence {abs(v2):.6f} should be ~0 for balanced input"
        assert abs(v0) < 0.05, f"Zero sequence {abs(v0):.6f} should be ~0 for balanced input"

    def test_symmetrical_components_slg_fault(self):
        """
        For SLG fault (Va=0, Vb and Vc balanced), negative and zero sequence
        should be non-negligible relative to positive sequence.
        """
        # Simulate SLG: phase A collapses
        va = complex(0, 0)
        vb = np.exp(-1j * 2 * np.pi / 3)
        vc = np.exp(1j * 2 * np.pi / 3)

        v0, v1, v2 = _symmetrical_components(va, vb, vc)

        # For this fault, V1 ≈ V2 ≈ V0 (Fortescue analysis of SLG)
        assert abs(v2) > 0.1, f"Negative sequence {abs(v2):.3f} too low for SLG fault"
        assert abs(v0) > 0.1, f"Zero sequence {abs(v0):.3f} too low for SLG fault"

    def test_batch_transform(self):
        """Batch transform must produce correct (n_windows, n_features) shape."""
        rng = np.random.default_rng(42)
        n_windows = 20
        windows = rng.standard_normal((n_windows, self.n_ch, self.window_size)).astype(np.float32)
        X = self.extractor.transform(windows)

        assert X.ndim == 2
        assert X.shape[0] == n_windows
        assert X.shape[1] > 0
        assert not np.isnan(X).any()

    def test_feature_names_match_vector_length(self):
        """Feature names list must match the extracted vector length."""
        win = self._make_window()
        feats = self.extractor.extract_window(win)
        names = self.extractor.get_feature_names()
        assert len(names) == len(feats), \
            f"Feature names ({len(names)}) ≠ feature vector length ({len(feats)})"

    def test_rocof_near_zero_for_stable_freq(self):
        """ROCOF should be near zero for a stable frequency signal."""
        freq_stable = np.full(self.window_size, 50.0, dtype=np.float32)
        rocof = self.extractor._rocof(freq_stable)
        assert rocof < 0.01, f"ROCOF={rocof:.4f} too high for stable 50 Hz signal"

    def test_rocof_high_for_ramp(self):
        """ROCOF should be non-zero for a ramping frequency signal."""
        freq_ramp = np.linspace(49.0, 51.0, self.window_size).astype(np.float32)
        rocof = self.extractor._rocof(freq_ramp)
        assert rocof > 0.1, f"ROCOF={rocof:.4f} too low for ramp frequency event"
