"""
Data loading and schema validation.

Loads raw or processed CSV files, validates schema, and returns
typed DataFrames ready for the preprocessing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

CHANNEL_NAMES = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Freq", "PF", "THD_V", "In"]
LABEL_COL = "label"
REQUIRED_COLS = CHANNEL_NAMES + [LABEL_COL]


class DataLoader:
    """Loads and validates power system fault datasets from CSV."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = cfg["general"]["random_seed"]

    def load(self, path: Path) -> pd.DataFrame:
        """
        Load CSV and validate schema.

        Args:
            path: Path to CSV file.

        Returns:
            Validated DataFrame with correct dtypes.

        Raises:
            ValueError: If required columns are missing or dtypes are wrong.
        """
        logger.info("Loading data from %s", path)
        df = pd.read_csv(path)
        self._validate_schema(df)
        df[CHANNEL_NAMES] = df[CHANNEL_NAMES].astype(np.float32)
        df[LABEL_COL] = df[LABEL_COL].astype(np.int32)
        logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path.name)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Raise if required columns are absent."""
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df[LABEL_COL].isnull().any():
            raise ValueError("Null values found in label column")

    def load_splits(self, synthetic_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load pre-saved train/val/test split CSVs from the synthetic data directory.

        Returns:
            (train_df, val_df, test_df)
        """
        train = self.load(synthetic_dir / "train.csv")
        val = self.load(synthetic_dir / "val.csv")
        test = self.load(synthetic_dir / "test.csv")
        return train, val, test

    def to_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and label vector y from DataFrame."""
        X = df[CHANNEL_NAMES].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.int32)
        return X, y

    def summary(self, df: pd.DataFrame) -> None:
        """Log class distribution summary."""
        counts = df[LABEL_COL].value_counts().sort_index()
        total = len(df)
        logger.info("Class distribution (total=%d):", total)
        for cls_id, cnt in counts.items():
            logger.info("  Class %d: %d (%.1f%%)", cls_id, cnt, 100 * cnt / total)
