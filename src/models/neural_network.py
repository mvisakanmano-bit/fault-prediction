"""
Temporal CNN for power system fault classification (PyTorch).

Architecture: three Conv1D blocks → AdaptiveAvgPool → two FC layers.
Input is raw windowed sensor signals (channel-first), not feature vectors,
allowing the network to learn its own temporal representations.

Design choices:
  - AdamW + CosineAnnealingLR for stable convergence
  - CrossEntropyLoss with class weights for imbalanced classes
  - Mixed precision (torch.cuda.amp) with CPU fallback
  - Gradient clipping at 1.0 to prevent exploding gradients
  - Best checkpoint saved by val F1-macro (not val loss)
  - Early stopping patience=15 epochs
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics import f1_score

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class TemporalCNN(nn.Module):
    """
    Three-stage temporal convolutional network for fault classification.

    Input:  (batch, n_channels=10, window_size=256)
    Output: (batch, n_classes=7) logits
    """

    def __init__(self, n_input_channels: int, n_classes: int, cfg: dict):
        super().__init__()
        nn_cfg = cfg["models"]["neural_network"]
        conv_ch = nn_cfg["conv_channels"]       # [64, 128, 256]
        kernels = nn_cfg["kernel_sizes"]         # [7, 5, 3]
        fc_units = nn_cfg["fc_units"]            # [128, 64]
        dropouts = nn_cfg["dropout_rates"]       # [0.4, 0.3]

        # Conv block builder
        def conv_block(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            )

        self.conv_blocks = nn.Sequential(
            conv_block(n_input_channels, conv_ch[0], kernels[0]),
            conv_block(conv_ch[0], conv_ch[1], kernels[1]),
            conv_block(conv_ch[1], conv_ch[2], kernels[2]),
            nn.AdaptiveAvgPool1d(1),   # Global temporal pooling → (batch, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_ch[2], fc_units[0]),
            nn.Dropout(dropouts[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_units[0], fc_units[1]),
            nn.Dropout(dropouts[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_units[1], n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.conv_blocks(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class NeuralNetworkModel(BaseModel):
    """
    PyTorch Temporal CNN wrapper implementing the BaseModel interface.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. See https://pytorch.org/get-started/locally/")

        nn_cfg = cfg["models"]["neural_network"]
        self._lr = nn_cfg["learning_rate"]
        self._weight_decay = nn_cfg["weight_decay"]
        self._batch_size = nn_cfg["batch_size"]
        self._epochs = nn_cfg["epochs"]
        self._patience = nn_cfg["early_stopping_patience"]
        self._grad_clip = nn_cfg["grad_clip_max_norm"]
        self._t_max = nn_cfg["scheduler_t_max"]
        self._use_amp = nn_cfg["mixed_precision"]
        self._seed = cfg["general"]["random_seed"]
        self._n_classes = cfg["data_generation"]["num_classes"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("NeuralNetwork: using device=%s", self.device)

        # Populated during train()
        self._network: Optional[TemporalCNN] = None
        self._n_channels: int = 10   # matches CHANNEL_NAMES length

    def _build_network(self) -> TemporalCNN:
        torch.manual_seed(self._seed)
        return TemporalCNN(self._n_channels, self._n_classes, self.cfg).to(self.device)

    def _compute_class_weights(self, y: np.ndarray) -> "torch.Tensor":
        """Inverse-frequency class weights for CrossEntropyLoss."""
        counts = np.bincount(y.astype(int), minlength=self._n_classes).astype(float)
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self._n_classes   # normalize
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> "DataLoader":
        """
        X must be (n_samples, n_channels, window_size) — channel-first.
        If X is 2D feature matrix, treat as (n_samples, n_features, 1) — handles both paths.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self._batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """
        Full training loop with:
          - AdamW optimizer
          - CosineAnnealingLR scheduler
          - Mixed precision (if GPU available)
          - Gradient clipping
          - Early stopping on val F1-macro
          - Best checkpoint saved to a temp file, restored at end
        """
        # Infer channel count from data shape
        if X_train.ndim == 3:
            self._n_channels = X_train.shape[1]
        else:
            # Flat feature vector; reshape to (n, n_feats, 1) for 1D conv
            X_train = X_train[:, :, np.newaxis]
            X_val = X_val[:, :, np.newaxis]
            self._n_channels = X_train.shape[1]

        self._network = self._build_network()

        class_weights = self._compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            self._network.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._t_max)

        # Mixed precision scaler (only meaningful on CUDA)
        use_amp = self._use_amp and self.device.type == "cuda"
        # torch.amp.GradScaler is the non-deprecated API (torch 2.1+)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # fallback

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_f1 = -1.0
        patience_counter = 0
        best_ckpt_path = Path(tempfile.mktemp(suffix=".pt"))

        logger.info(
            "NeuralNetwork: training for up to %d epochs (patience=%d)...",
            self._epochs, self._patience,
        )

        for epoch in range(1, self._epochs + 1):
            # --- Training phase ---
            self._network.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self._network(batch_X)
                    loss = criterion(logits, batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self._network.parameters(), self._grad_clip)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * len(batch_y)

            scheduler.step()
            train_loss /= len(train_loader.dataset)

            # --- Validation phase ---
            val_f1, val_loss = self._evaluate(val_loader, criterion)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(self._network.state_dict(), best_ckpt_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == 0:
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | patience=%d",
                    epoch, self._epochs, train_loss, val_loss, val_f1, patience_counter,
                )

            if patience_counter >= self._patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        # Restore best checkpoint
        self._network.load_state_dict(torch.load(best_ckpt_path, map_location=self.device))
        best_ckpt_path.unlink(missing_ok=True)
        self.is_fitted = True

        logger.info("NeuralNetwork training complete. Best val F1-macro=%.4f", best_val_f1)
        return {"best_val_f1": best_val_f1}

    def _evaluate(
        self,
        loader: "DataLoader",
        criterion: "nn.Module",
    ) -> Tuple[float, float]:
        """Compute val F1-macro and val loss over the given loader."""
        self._network.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self._network(batch_X)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_y)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return float(f1), float(avg_loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self._network is None:
            raise RuntimeError("Model not fitted. Call train() first.")

        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        self._network.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self._batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                logits = self._network(batch_X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._network.state_dict(),
            "n_channels": self._n_channels,
            "n_classes": self._n_classes,
            "cfg": self.cfg,
        }, path)
        logger.info("NeuralNetwork saved to %s", path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self._n_channels = payload["n_channels"]
        self._n_classes = payload["n_classes"]
        self._network = self._build_network()
        self._network.load_state_dict(payload["state_dict"])
        self._network.eval()
        self.is_fitted = True
        logger.info("NeuralNetwork loaded from %s", path)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """NN does not expose simple feature importances. Use Grad-CAM instead."""
        return None
