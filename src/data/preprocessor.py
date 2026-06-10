"""
Data preprocessing pipeline: missing value handling, outlier clipping,
RobustScaler normalization, and sliding window extraction.

Implemented as an sklearn-compatible Pipeline for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import joblib

logger = logging.getLogger(__name__)

CHANNEL_NAMES = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Freq", "PF", "THD_V", "In"]


# ---------------------------------------------------------------------------
# Custom sklearn transformers
# ---------------------------------------------------------------------------

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Forward-fill gaps <= max_fill samples; drop rows in segments with
    longer consecutive runs of NaN.
    """

    def __init__(self, max_fill: int = 5):
        self.max_fill = max_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        df = pd.DataFrame(X)
        # Forward-fill short gaps (pandas 2.2+ uses ffill() instead of fillna(method=))
        df = df.ffill(limit=self.max_fill)
        # Drop any remaining NaN rows (long gaps)
        n_dropped = df.isnull().any(axis=1).sum()
        if n_dropped > 0:
            logger.warning("Dropping %d rows with long NaN segments (>%d consecutive)", n_dropped, self.max_fill)
        df = df.dropna()
        return df.values.astype(np.float32)


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Per-channel IQR-based outlier clipping at [Q1 - k*IQR, Q3 + k*IQR].

    Fitted on training data; transforms new data using fitted bounds.
    Fault spikes that represent real events will be partially clipped,
    but the RobustScaler downstream handles remaining scale variation.
    """

    def __init__(self, k: float = 3.0):
        self.k = k
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "IQRClipper":
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        logger.debug("IQRClipper fitted: lower=%s, upper=%s", self.lower_, self.upper_)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return np.clip(X, self.lower_, self.upper_).astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(cfg: dict) -> Pipeline:
    """
    Build and return the sklearn preprocessing pipeline.

    Steps:
        1. MissingValueHandler — forward-fill short gaps
        2. IQRClipper — clip outliers per channel
        3. RobustScaler — median/IQR-based normalization

    Args:
        cfg: Full project config dict.

    Returns:
        Unfitted sklearn Pipeline.
    """
    prep_cfg = cfg["preprocessing"]
    return Pipeline([
        ("missing", MissingValueHandler(max_fill=prep_cfg["missing_fill_max_samples"])),
        ("clipper", IQRClipper(k=prep_cfg["outlier_iqr_multiplier"])),
        ("scaler", RobustScaler()),
    ])


# ---------------------------------------------------------------------------
# Sliding window extraction
# ---------------------------------------------------------------------------

def sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping sliding windows from a time-series array.

    Args:
        X: (n_samples, n_channels) array of sensor data.
        y: (n_samples,) array of labels.
        window_size: Number of samples per window (e.g., 256).
        stride: Step between window starts (e.g., 64 for 75% overlap).

    Returns:
        windows: (n_windows, n_channels, window_size) — channel-first for CNN.
        labels: (n_windows,) — majority-vote label per window.
    """
    n_samples, n_channels = X.shape
    n_windows = max(0, (n_samples - window_size) // stride + 1)

    windows = np.empty((n_windows, n_channels, window_size), dtype=np.float32)
    labels = np.empty(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = X[start:end].T          # transpose: (window_size, n_ch) → (n_ch, window_size)
        # Majority vote for label; fault labels dominate if present
        label_slice = y[start:end]
        labels[i] = np.bincount(label_slice.astype(int)).argmax()

    logger.debug(
        "Sliding windows: %d samples → %d windows (size=%d, stride=%d)",
        n_samples, n_windows, window_size, stride,
    )
    return windows, labels


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Wraps the sklearn pipeline and sliding window extraction into
    a single fit/transform interface for use by the training loop.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.window_size = cfg["preprocessing"]["window_size"]
        self.stride = cfg["preprocessing"]["stride"]
        self.pipeline: Optional[Pipeline] = None

    def fit_transform(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
        """
        Fit pipeline on X_raw, transform, then extract sliding windows.

        Returns:
            windows, labels, fitted_pipeline
        """
        self.pipeline = build_preprocessing_pipeline(self.cfg)
        X_scaled = self.pipeline.fit_transform(X_raw)
        windows, labels = sliding_windows(X_scaled, y, self.window_size, self.stride)
        logger.info(
            "fit_transform: raw=%s → scaled=%s → windows=%s",
            X_raw.shape, X_scaled.shape, windows.shape,
        )
        return windows, labels, self.pipeline

    def transform(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fitted pipeline to new data and extract windows.
        Must call fit_transform first.
        """
        if self.pipeline is None:
            raise RuntimeError("Call fit_transform before transform")
        X_scaled = self.pipeline.transform(X_raw)
        windows, labels = sliding_windows(X_scaled, y, self.window_size, self.stride)
        return windows, labels

    def save_pipeline(self, path: Path) -> None:
        """Serialize fitted pipeline to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Preprocessing pipeline saved to %s", path)

    @classmethod
    def load_pipeline(cls, path: Path) -> Pipeline:
        """Load a previously fitted pipeline."""
        return joblib.load(path)
