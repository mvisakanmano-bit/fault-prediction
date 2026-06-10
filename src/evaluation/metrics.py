"""
Comprehensive evaluation metrics for power system fault classification.

Implements all standard classification metrics plus fault-detection-specific
safety metrics required for protection relay contexts:
  - FDR (Fault Detection Rate) — must be >95% per class
  - FAR (False Alarm Rate) — must be <2% per class
  - MTTD (Mean Time to Detection) — relay operating time equivalent
  - ECE (Expected Calibration Error) — probability trustworthiness
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

logger = logging.getLogger(__name__)

FAULT_NAMES = ["Normal", "SLG", "LL", "3PH", "Overload", "VoltageSag", "Harmonic"]


class FaultMetrics:
    """
    Compute and store all evaluation metrics for one model on one dataset split.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n_classes = cfg["data_generation"]["num_classes"]

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        window_stride: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute the full suite of metrics.

        Args:
            y_true:  (n,) ground-truth integer labels.
            y_pred:  (n,) predicted integer labels.
            y_proba: (n, n_classes) predicted probability matrix.
            window_stride: Stride in samples between windows (for MTTD computation).
            fs: Sample rate in Hz (for MTTD in ms).

        Returns:
            Dict of all metric values; nested dicts for per-class metrics.
        """
        if window_stride is None:
            window_stride = self.cfg["preprocessing"]["stride"]
        if fs is None:
            fs = self.cfg["data_generation"]["sample_rate"]

        metrics: Dict[str, Any] = {}

        # --- Standard classification metrics ---
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # Per-class F1, precision, recall
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics["per_class_f1"] = {FAULT_NAMES[i]: float(v) for i, v in enumerate(per_class_f1)}
        metrics["per_class_precision"] = {FAULT_NAMES[i]: float(v) for i, v in enumerate(per_class_prec)}
        metrics["per_class_recall"] = {FAULT_NAMES[i]: float(v) for i, v in enumerate(per_class_rec)}

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.n_classes)))
        metrics["confusion_matrix"] = cm.tolist()
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        metrics["confusion_matrix_normalized"] = cm_norm.tolist()

        # --- ROC-AUC and PR-AUC (One-vs-Rest) ---
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")

        per_class_pr_auc = {}
        for cls_id in range(self.n_classes):
            y_bin = (y_true == cls_id).astype(int)
            prob_cls = y_proba[:, cls_id]
            try:
                per_class_pr_auc[FAULT_NAMES[cls_id]] = float(average_precision_score(y_bin, prob_cls))
            except Exception:
                per_class_pr_auc[FAULT_NAMES[cls_id]] = float("nan")
        metrics["per_class_pr_auc"] = per_class_pr_auc

        # --- Fault-detection-specific metrics ---
        fdr_per_class, far_per_class = self._compute_fdr_far(cm)
        metrics["per_class_fdr"] = {FAULT_NAMES[i]: float(v) for i, v in enumerate(fdr_per_class)}
        metrics["per_class_far"] = {FAULT_NAMES[i]: float(v) for i, v in enumerate(far_per_class)}
        # Exclude Normal class (class 0) from fault detection averages
        fault_fdr = [fdr_per_class[i] for i in range(1, self.n_classes)]
        fault_far = [far_per_class[i] for i in range(1, self.n_classes)]
        metrics["fdr_mean"] = float(np.nanmean(fault_fdr))
        metrics["far_mean"] = float(np.nanmean(fault_far))

        # MTTD (Mean Time to Detection in milliseconds)
        mttd_ms = self._compute_mttd(y_true, y_pred, window_stride, fs)
        metrics["mttd_ms"] = mttd_ms

        # --- Calibration metrics ---
        ece = self._compute_ece(y_true, y_proba)
        metrics["ece"] = ece

        # Brier score (per class, one-vs-rest)
        brier_scores = {}
        for cls_id in range(self.n_classes):
            y_bin = (y_true == cls_id).astype(float)
            prob_cls = y_proba[:, cls_id]
            brier_scores[FAULT_NAMES[cls_id]] = float(brier_score_loss(y_bin, prob_cls))
        metrics["per_class_brier"] = brier_scores
        metrics["brier_mean"] = float(np.mean(list(brier_scores.values())))

        return metrics

    def _compute_fdr_far(
        self, cm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-class Fault Detection Rate and False Alarm Rate from confusion matrix.

        FDR (class i) = TP_i / (TP_i + FN_i) = cm[i,i] / sum(cm[i,:])
        FAR (class i) = FP_i / (FP_i + TN_i)
                      = sum(cm[:,i]) - cm[i,i]) / (total - sum(cm[i,:]))

        FDR corresponds to sensitivity/recall per class.
        FAR corresponds to fall-out rate per class — a nuisance trip metric.
        """
        n = cm.shape[0]
        fdr = np.zeros(n)
        far = np.zeros(n)
        total = cm.sum()

        for i in range(n):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp         # actual class i predicted as something else
            fp = cm[:, i].sum() - tp         # other classes predicted as class i
            tn = total - tp - fn - fp

            fdr[i] = tp / max(tp + fn, 1)
            far[i] = fp / max(fp + tn, 1)

        return fdr, far

    def _compute_mttd(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        window_stride: int,
        fs: int,
    ) -> float:
        """
        Compute Mean Time to Detection (MTTD) in milliseconds.

        For each fault event (transition from class 0 to any fault class),
        counts the number of windows until the first correct prediction.
        The detection lag in windows × stride / fs gives time in seconds.

        Returns mean detection lag in milliseconds across all fault events,
        or NaN if no fault events are found.
        """
        detection_lags_ms = []
        in_fault = False
        fault_start = None
        current_fault_class = None

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if not in_fault and true != 0:
                # Fault onset
                in_fault = True
                fault_start = i
                current_fault_class = int(true)

            elif in_fault:
                if int(pred) == current_fault_class:
                    # First correct detection
                    lag_windows = i - fault_start
                    lag_ms = lag_windows * window_stride / fs * 1000
                    detection_lags_ms.append(lag_ms)
                    in_fault = False
                    fault_start = None

                elif int(true) == 0:
                    # Fault cleared without detection (miss)
                    in_fault = False
                    fault_start = None

        if detection_lags_ms:
            return float(np.mean(detection_lags_ms))
        return float("nan")

    def _compute_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """
        Expected Calibration Error (ECE).

        Bins predictions by their maximum predicted probability and measures
        the average gap between mean predicted probability and actual accuracy
        in each bin. Lower is better; 0 = perfect calibration.
        """
        max_probs = y_proba.max(axis=1)
        pred_labels = y_proba.argmax(axis=1)
        correct = (pred_labels == y_true).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for b in range(n_bins):
            mask = (max_probs >= bin_edges[b]) & (max_probs < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = max_probs[mask].mean()
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

        return float(ece)

    def print_report(self, metrics: Dict[str, Any], model_name: str = "") -> None:
        """Print a human-readable metrics summary."""
        header = f"METRICS — {model_name}" if model_name else "METRICS"
        print("\n" + "=" * 65)
        print(header)
        print("=" * 65)

        scalar_keys = [
            "accuracy", "f1_macro", "f1_weighted", "precision_macro",
            "recall_macro", "mcc", "cohen_kappa", "roc_auc_ovr",
            "fdr_mean", "far_mean", "mttd_ms", "ece", "brier_mean",
        ]
        for k in scalar_keys:
            v = metrics.get(k, "N/A")
            if isinstance(v, float) and not (v != v):  # NaN check
                print(f"  {k:<30s}: {v:.4f}")
            else:
                print(f"  {k:<30s}: {v}")

        print("\n  Per-Class FDR:")
        for cls, val in metrics.get("per_class_fdr", {}).items():
            bar = "✓" if val >= 0.95 else "✗"
            print(f"    {cls:<16s}: {val:.3f} {bar}")

        print("\n  Per-Class FAR:")
        for cls, val in metrics.get("per_class_far", {}).items():
            bar = "✓" if val <= 0.02 else "✗"
            print(f"    {cls:<16s}: {val:.3f} {bar}")

        print("=" * 65)
