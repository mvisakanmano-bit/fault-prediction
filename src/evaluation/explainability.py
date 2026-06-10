"""
Explainability visualizations for trained fault prediction models.

Generates:
  - SHAP beeswarm summary plots (RF, XGBoost)
  - SHAP waterfall plots per fault class
  - Grad-CAM for Neural Network (via conv layer hooks)
  - Feature correlation heatmap (top 30 features)
  - HTML evaluation report
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

FAULT_NAMES = ["Normal", "SLG", "LL", "3PH", "Overload", "VoltageSag", "Harmonic"]


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", path)


# ---------------------------------------------------------------------------
# SHAP visualizations
# ---------------------------------------------------------------------------

def plot_shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    dpi: int = 300,
    max_display: int = 20,
) -> None:
    """
    SHAP beeswarm summary plot for RF or XGBoost.

    Each dot represents one prediction; position on x-axis shows SHAP value
    (impact on model output), color shows feature value magnitude.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping summary plot")
        return

    # Subsample for speed (SHAP can be slow on large datasets)
    n_samples = min(500, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    X_sub = X[idx]

    try:
        # Try TreeExplainer first (fast for RF/XGBoost)
        explainer = shap.TreeExplainer(model.model if hasattr(model, "model") else model)
        shap_values = explainer.shap_values(X_sub)
    except Exception:
        # Fall back to KernelExplainer (model-agnostic, slower)
        background = shap.sample(X, 100)
        explainer = shap.KernelExplainer(
            model.predict_proba if hasattr(model, "predict_proba") else model,
            background,
        )
        shap_values = explainer.shap_values(X_sub)

    # For multi-class, shap_values is a list; use the mean absolute across classes
    if isinstance(shap_values, list):
        shap_matrix = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        shap_matrix = shap_values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_matrix, X_sub,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Importance (mean |SHAP| across classes)", fontsize=12)
    plt.tight_layout()
    _save_fig(plt.gcf(), output_path, dpi=dpi)


def plot_shap_waterfall(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """
    SHAP waterfall plots for one representative sample per fault class.
    Shows contribution of each feature to pushing the prediction toward
    the true class.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping waterfall plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        explainer = shap.TreeExplainer(model.model if hasattr(model, "model") else model)
    except Exception as e:
        logger.warning("Cannot create SHAP explainer for waterfall: %s", e)
        return

    for cls_id, cls_name in enumerate(FAULT_NAMES):
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue

        # Select first matching sample
        sample_idx = np.where(mask)[0][0]
        sample = X[sample_idx : sample_idx + 1]

        try:
            shap_vals = explainer.shap_values(sample)
            if isinstance(shap_vals, list):
                # Use SHAP values for the true class
                sv = shap_vals[cls_id][0]
            else:
                sv = shap_vals[0]

            expected = explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = expected[cls_id]

            # Manual waterfall-style bar chart (shap.waterfall_plot API varies by version)
            top_n = 15
            abs_sv = np.abs(sv)
            top_idx = np.argsort(abs_sv)[::-1][:top_n]
            top_names = [feature_names[i] for i in top_idx]
            top_vals = sv[top_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_vals]
            ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_names[::-1], fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on log-odds)")
            ax.set_title(f"SHAP Waterfall — Class {cls_id}: {cls_name}", fontsize=12)
            plt.tight_layout()

            _save_fig(fig, output_dir / f"shap_waterfall_class{cls_id}_{cls_name}.png", dpi=dpi)

        except Exception as e:
            logger.warning("SHAP waterfall failed for class %d: %s", cls_id, e)


# ---------------------------------------------------------------------------
# Grad-CAM for Neural Network
# ---------------------------------------------------------------------------

def plot_gradcam(
    model: Any,
    X_windows: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """
    Grad-CAM visualization for the Temporal CNN.

    Hooks into the final conv block's output to compute gradient-weighted
    activation maps, showing which time steps most influenced the prediction.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.warning("PyTorch not available — skipping Grad-CAM")
        return

    if not hasattr(model, "_network") or model._network is None:
        logger.warning("Neural network not loaded — skipping Grad-CAM")
        return

    network = model._network
    network.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Register hook on the last conv block (index 2 in conv_blocks Sequential)
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["last_conv"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["last_conv"] = grad_out[0].detach()

    # The last MaxPool1d is at conv_blocks[-2], last conv block at conv_blocks[2]
    # We hook onto the ReLU of the third conv block
    try:
        target_layer = network.conv_blocks[2][2]  # ReLU in 3rd conv block
        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)
    except (IndexError, AttributeError) as e:
        logger.warning("Grad-CAM hook registration failed: %s", e)
        return

    device = model.device

    for cls_id, cls_name in enumerate(FAULT_NAMES):
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue

        sample_idx = np.where(mask)[0][0]
        x = X_windows[sample_idx : sample_idx + 1]
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        x_t.requires_grad_(True)

        logits = network(x_t)
        score = logits[0, cls_id]
        network.zero_grad()
        score.backward()

        act = activations.get("last_conv")  # (1, C, T)
        grad = gradients.get("last_conv")   # (1, C, T)

        if act is None or grad is None:
            continue

        # Global average pooling of gradients → weights
        weights = grad.mean(dim=2, keepdim=True)     # (1, C, 1)
        cam = (weights * act).sum(dim=1).squeeze()   # (T,)
        cam = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Grad-CAM — Class {cls_id}: {cls_name}", fontsize=12)

        # Plot Va signal
        axes[0].plot(x[0, 0], label="Va (scaled)", alpha=0.8)
        axes[0].set_ylabel("Signal amplitude")
        axes[0].legend()

        # Plot CAM heatmap
        t_cam = np.linspace(0, len(x[0, 0]), len(cam))
        axes[1].fill_between(t_cam, cam, alpha=0.7, color="red", label="Grad-CAM")
        axes[1].set_ylabel("Activation importance")
        axes[1].set_xlabel("Time steps (within window)")
        axes[1].legend()

        plt.tight_layout()
        _save_fig(fig, output_dir / f"gradcam_class{cls_id}_{cls_name}.png", dpi=dpi)

    fwd_handle.remove()
    bwd_handle.remove()


# ---------------------------------------------------------------------------
# Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_feature_correlation(
    X: np.ndarray,
    feature_names: List[str],
    importances: Optional[np.ndarray],
    output_path: Path,
    top_n: int = 30,
    dpi: int = 300,
) -> None:
    """
    Correlation heatmap for the top_n most important features.

    Args:
        X: (n_samples, n_features) feature matrix.
        feature_names: Feature name list.
        importances: Feature importance array for selecting top_n features.
        output_path: Save path.
        top_n: Number of top features to show.
    """
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed — skipping correlation heatmap")
        return

    if importances is not None and len(importances) == len(feature_names):
        top_idx = np.argsort(importances)[::-1][:top_n]
    else:
        top_idx = np.arange(min(top_n, len(feature_names)))

    X_top = X[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]

    corr = np.corrcoef(X_top.T)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        xticklabels=names_top,
        yticklabels=names_top,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        ax=ax,
        square=True,
        linewidths=0.3,
    )
    ax.set_title(f"Feature Correlation Matrix (top {top_n} by importance)", fontsize=12)
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7, rotation=0)
    plt.tight_layout()
    _save_fig(fig, output_path, dpi=dpi)


# ---------------------------------------------------------------------------
# HTML evaluation report
# ---------------------------------------------------------------------------

def generate_html_report(
    all_results: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    output_path: Path,
) -> None:
    """
    Generate an HTML evaluation report with a comparative metrics table.

    Args:
        all_results: Dict mapping model_name → metrics dict.
        feature_names: List of feature names.
        output_path: Path for the .html file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scalar_metrics = [
        "accuracy", "f1_macro", "f1_weighted", "precision_macro",
        "recall_macro", "mcc", "cohen_kappa", "roc_auc_ovr",
        "fdr_mean", "far_mean", "mttd_ms", "ece", "brier_mean",
    ]

    model_names = list(all_results.keys())

    # Build table rows
    rows_html = ""
    for metric in scalar_metrics:
        row = f"<tr><td><b>{metric}</b></td>"
        vals = []
        for m in model_names:
            v = all_results[m].get(metric, float("nan"))
            if isinstance(v, float):
                vals.append(v)
                row += f"<td>{v:.4f}</td>"
            else:
                vals.append(None)
                row += f"<td>{v}</td>"
        row += "</tr>"
        rows_html += row

    # Per-class FDR/FAR table
    per_class_html = ""
    for cls_name in FAULT_NAMES:
        row = f"<tr><td>{cls_name}</td>"
        for m in model_names:
            fdr = all_results[m].get("per_class_fdr", {}).get(cls_name, float("nan"))
            far = all_results[m].get("per_class_far", {}).get(cls_name, float("nan"))
            row += f"<td>FDR={fdr:.3f} / FAR={far:.3f}</td>"
        row += "</tr>"
        per_class_html += row

    headers = "".join(f"<th>{m}</th>" for m in model_names)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fault Prediction Evaluation Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; }}
h1 {{ color: #2c3e50; }}
h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background: white; }}
th, td {{ padding: 10px 14px; text-align: center; border: 1px solid #ddd; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.note {{ color: #7f8c8d; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>AI Fault Prediction — Evaluation Report</h1>
<p class="note">Generated by fault_prediction pipeline. Targets IEC 61850-compliant SCADA integration.</p>

<h2>Comparative Metrics</h2>
<table>
<thead><tr><th>Metric</th>{headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table>

<h2>Per-Class FDR / FAR</h2>
<p class="note">FDR must be &gt;95% | FAR must be &lt;2% for safety-critical deployment</p>
<table>
<thead><tr><th>Fault Class</th>{headers}</tr></thead>
<tbody>{per_class_html}</tbody>
</table>

<h2>Feature Count</h2>
<p>{len(feature_names)} features extracted per window.</p>

<h2>Figures</h2>
<p>See <code>outputs/plots/</code> for SHAP beeswarm, waterfall, Grad-CAM, and correlation heatmap figures.</p>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("HTML report saved: %s", output_path)
