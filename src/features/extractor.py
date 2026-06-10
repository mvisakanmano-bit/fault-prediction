"""
Feature extraction pipeline for power system fault classification.

Extracts ~80–120 features per sliding window across three domains:
  1. Statistical (time-domain) — per channel
  2. Frequency-domain (FFT-based) — harmonic content, THD, spectral entropy
  3. Power-system-specific — symmetrical components, ROCOF, imbalance ratios

Usage (validation gate):
    python -m src.features.extractor --test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.fft import fft, fftfreq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_config, setup_logging

logger = logging.getLogger(__name__)

# Channel indices within the (n_ch, window_size) array
CH = {
    "Va": 0, "Vb": 1, "Vc": 2,
    "Ia": 3, "Ib": 4, "Ic": 5,
    "Freq": 6, "PF": 7, "THD_V": 8, "In": 9,
}

# Clarke/Fortescue transform matrix (3-phase to symmetrical components)
# Alpha = e^{j*2π/3}
_ALPHA = np.exp(1j * 2 * np.pi / 3)
_A_MATRIX = np.array([
    [1,            1,            1           ],   # row 0: zero sequence
    [1,            _ALPHA,       _ALPHA ** 2 ],   # row 1: positive sequence (standard Fortescue)
    [1,            _ALPHA ** 2,  _ALPHA      ],   # row 2: negative sequence
], dtype=complex) / 3.0


def _symmetrical_components(va: np.ndarray, vb: np.ndarray, vc: np.ndarray) -> Tuple[complex, complex, complex]:
    """
    Compute zero, positive, and negative sequence components from RMS phasors.

    Uses the Fortescue transformation: [V0, V1, V2] = A * [Va, Vb, Vc]
    where A is the symmetric component transformation matrix.

    Critical for fault type discrimination: negative/zero sequence rise
    sharply during asymmetric faults while positive sequence dominates
    in balanced (normal/3-phase) conditions.

    Args:
        va, vb, vc: Complex phasor values (scalar) for each phase.

    Returns:
        (V0, V1, V2) — zero, positive, negative sequence phasors.
    """
    v_vec = np.array([va, vb, vc], dtype=complex)
    seq = _A_MATRIX @ v_vec
    return seq[0], seq[1], seq[2]   # V0, V1, V2


def _phasor_from_window(signal: np.ndarray, freq: float, fs: float) -> complex:
    """
    Estimate fundamental-frequency phasor from a time-domain window via DFT.

    Finds the DFT bin closest to `freq` Hz and returns the complex amplitude.
    """
    n = len(signal)
    spectrum = fft(signal, n=n)
    freqs = fftfreq(n, d=1.0 / fs)
    # Find positive-frequency bin closest to fundamental
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_spec = spectrum[pos_mask]
    idx = np.argmin(np.abs(pos_freqs - freq))
    # Scale by 2/n so magnitude equals peak amplitude, then convert to RMS phasor
    return pos_spec[idx] * 2 / n / np.sqrt(2)


class FeatureExtractor:
    """
    Extracts a fixed-length feature vector from each (n_channels, window_size) window.

    The extracted features are grouped into:
      - Statistical: mean, std, min, max, p2p, rms, skewness, kurtosis, zcr, mad
      - Frequency: dominant_freq, spectral_entropy, band powers (fund, 2nd–7th harmonic), THD
      - Power-system: symmetrical components (mag+angle), imbalance ratios, apparent power, ROCOF
      - Cross-channel: cross-correlation (Va–Ia), phase angle (per phase)

    Total: approximately 100–115 features.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        feat_cfg = cfg["features"]
        self.fs = cfg["data_generation"]["sample_rate"]   # 1000 Hz
        self.f_fund = feat_cfg["fundamental_freq"]        # 50 Hz
        self.harmonic_orders = feat_cfg["harmonics"]      # [2, 3, 5, 7]
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Per-channel statistical features
    # ------------------------------------------------------------------

    def _stat_features(self, x: np.ndarray, prefix: str) -> Dict[str, float]:
        """10 statistical descriptors for a single channel signal."""
        n = len(x)
        rms = float(np.sqrt(np.mean(x ** 2)))
        zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / n
        mad = float(np.mean(np.abs(x - np.mean(x))))
        sk = float(sp_stats.skew(x)) if np.std(x) > 1e-10 else 0.0
        ku = float(sp_stats.kurtosis(x)) if np.std(x) > 1e-10 else 0.0

        return {
            f"{prefix}_mean":  float(np.mean(x)),
            f"{prefix}_std":   float(np.std(x)),
            f"{prefix}_min":   float(np.min(x)),
            f"{prefix}_max":   float(np.max(x)),
            f"{prefix}_p2p":   float(np.ptp(x)),
            f"{prefix}_rms":   rms,
            f"{prefix}_skew":  sk,
            f"{prefix}_kurt":  ku,
            f"{prefix}_zcr":   zcr,
            f"{prefix}_mad":   mad,
        }

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------

    def _fft_features(self, x: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        FFT-based features for a single channel:
          - Dominant frequency
          - Spectral entropy (information content across frequency bins)
          - Power at fundamental and harmonic bands
          - THD computed from harmonic amplitudes
        """
        n = len(x)
        # Apply Hann window to reduce spectral leakage
        window = np.hanning(n)
        x_w = x * window

        spectrum = np.abs(fft(x_w, n=n))[:n // 2]
        freqs = fftfreq(n, d=1.0 / self.fs)[:n // 2]
        spectrum = spectrum * 2 / n  # scale to peak amplitude

        # Dominant frequency (ignore DC)
        pos_freqs = freqs[1:]
        pos_spec = spectrum[1:]
        dom_freq = float(pos_freqs[np.argmax(pos_spec)]) if len(pos_spec) > 0 else 0.0

        # Spectral entropy (Shannon entropy of normalized power spectrum)
        power = spectrum ** 2
        power_sum = power.sum()
        if power_sum > 1e-12:
            p_norm = power / power_sum
            p_norm = p_norm[p_norm > 1e-12]
            spec_entropy = float(-np.sum(p_norm * np.log2(p_norm)))
        else:
            spec_entropy = 0.0

        # Band power extraction: power within ±2 Hz of target frequency
        def band_power(target_hz: float) -> float:
            mask = np.abs(freqs - target_hz) <= 2.0
            return float(np.sum(spectrum[mask] ** 2))

        h1_power = band_power(self.f_fund)
        harmonic_powers = {
            f"{prefix}_h{order}_power": band_power(order * self.f_fund)
            for order in self.harmonic_orders
        }

        # THD from FFT: sqrt(sum(Hn^2)) / H1
        h1_amp = float(np.sqrt(h1_power)) if h1_power > 0 else 1e-10
        harmonic_amps = [float(np.sqrt(max(harmonic_powers[f"{prefix}_h{o}_power"], 0)))
                         for o in self.harmonic_orders]
        thd = float(np.sqrt(sum(a ** 2 for a in harmonic_amps)) / h1_amp * 100)

        feats = {
            f"{prefix}_dom_freq":      dom_freq,
            f"{prefix}_spec_entropy":  spec_entropy,
            f"{prefix}_h1_power":      h1_power,
            f"{prefix}_thd_fft":       thd,
        }
        feats.update(harmonic_powers)
        return feats

    # ------------------------------------------------------------------
    # Power-system-specific features
    # ------------------------------------------------------------------

    def _symmetrical_component_features(
        self,
        va_win: np.ndarray,
        vb_win: np.ndarray,
        vc_win: np.ndarray,
        ia_win: np.ndarray,
        ib_win: np.ndarray,
        ic_win: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute positive/negative/zero sequence magnitudes and angles
        for both voltage and current.

        Negative sequence rises for asymmetric faults (SLG, LL).
        Zero sequence is non-zero only when neutral current flows (SLG).
        Positive sequence dominant in balanced conditions.
        """
        # Estimate fundamental phasors via DFT
        phasors_v = [_phasor_from_window(s, self.f_fund, self.fs) for s in (va_win, vb_win, vc_win)]
        phasors_i = [_phasor_from_window(s, self.f_fund, self.fs) for s in (ia_win, ib_win, ic_win)]

        v0, v1, v2 = _symmetrical_components(*phasors_v)
        i0, i1, i2 = _symmetrical_components(*phasors_i)

        feats = {}
        for name, val in [("v0", v0), ("v1", v1), ("v2", v2), ("i0", i0), ("i1", i1), ("i2", i2)]:
            feats[f"seq_{name}_mag"] = float(np.abs(val))
            feats[f"seq_{name}_ang"] = float(np.angle(val))

        # Imbalance ratios: negative / positive sequence
        v1_mag = feats["seq_v1_mag"]
        i1_mag = feats["seq_i1_mag"]
        feats["v_neg_imb_ratio"] = feats["seq_v2_mag"] / (v1_mag + 1e-10)
        feats["i_neg_imb_ratio"] = feats["seq_i2_mag"] / (i1_mag + 1e-10)
        feats["v_zero_imb_ratio"] = feats["seq_v0_mag"] / (v1_mag + 1e-10)

        return feats

    def _power_features(
        self,
        va: np.ndarray, vb: np.ndarray, vc: np.ndarray,
        ia: np.ndarray, ib: np.ndarray, ic: np.ndarray,
    ) -> Dict[str, float]:
        """Apparent power per phase and voltage/current imbalance ratios."""
        def rms(x): return float(np.sqrt(np.mean(x ** 2)))

        va_rms, vb_rms, vc_rms = rms(va), rms(vb), rms(vc)
        ia_rms, ib_rms, ic_rms = rms(ia), rms(ib), rms(ic)

        sa = va_rms * ia_rms
        sb = vb_rms * ib_rms
        sc = vc_rms * ic_rms

        v_vals = np.array([va_rms, vb_rms, vc_rms])
        i_vals = np.array([ia_rms, ib_rms, ic_rms])
        v_mean = float(v_vals.mean()) if v_vals.mean() > 0 else 1e-10
        i_mean = float(i_vals.mean()) if i_vals.mean() > 0 else 1e-10

        # Imbalance ratio: (max - min) / mean — classic power quality metric
        v_imb = float((v_vals.max() - v_vals.min()) / v_mean)
        i_imb = float((i_vals.max() - i_vals.min()) / i_mean)

        return {
            "Sa": sa, "Sb": sb, "Sc": sc,
            "S_total": sa + sb + sc,
            "Va_rms": va_rms, "Vb_rms": vb_rms, "Vc_rms": vc_rms,
            "Ia_rms": ia_rms, "Ib_rms": ib_rms, "Ic_rms": ic_rms,
            "V_imbalance": v_imb,
            "I_imbalance": i_imb,
        }

    def _rocof(self, freq_win: np.ndarray) -> float:
        """
        Rate of Change of Frequency (ROCOF): df/dt via finite difference.

        Sensitive to inertia events preceding major frequency-affecting faults.
        Returns RMS of the ROCOF signal across the window (captures transient changes).
        """
        dt = 1.0 / self.fs
        df_dt = np.diff(freq_win) / dt
        return float(np.sqrt(np.mean(df_dt ** 2)))  # RMS ROCOF (Hz/s)

    def _cross_channel_features(
        self,
        va: np.ndarray, vb: np.ndarray, vc: np.ndarray,
        ia: np.ndarray, ib: np.ndarray, ic: np.ndarray,
    ) -> Dict[str, float]:
        """
        Cross-correlation between voltage and current per phase.
        Normally near zero for 50 Hz synchronized signals (controlled correlation).
        Spikes significantly during faults due to phase and magnitude shifts.
        """
        def xcorr_peak(x: np.ndarray, y: np.ndarray) -> float:
            """Peak normalized cross-correlation."""
            x_n = x - x.mean()
            y_n = y - y.mean()
            denom = np.sqrt((x_n ** 2).sum() * (y_n ** 2).sum())
            if denom < 1e-12:
                return 0.0
            corr = np.correlate(x_n, y_n, mode="full")
            return float(np.max(np.abs(corr)) / denom)

        def phase_angle(v: np.ndarray, i: np.ndarray) -> float:
            """Approximate phase angle between V and I using cross-correlation lag."""
            n = len(v)
            corr = np.correlate(v - v.mean(), i - i.mean(), mode="full")
            lag = int(np.argmax(np.abs(corr)) - (n - 1))
            # Convert lag in samples to angle in degrees
            samples_per_cycle = self.fs / self.f_fund
            return float(lag / samples_per_cycle * 360.0)

        return {
            "xcorr_Va_Ia": xcorr_peak(va, ia),
            "xcorr_Vb_Ib": xcorr_peak(vb, ib),
            "xcorr_Vc_Ic": xcorr_peak(vc, ic),
            "phase_angle_a": phase_angle(va, ia),
            "phase_angle_b": phase_angle(vb, ib),
            "phase_angle_c": phase_angle(vc, ic),
        }

    # ------------------------------------------------------------------
    # Main extraction method
    # ------------------------------------------------------------------

    def extract_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single window.

        Args:
            window: (n_channels, window_size) array — channel-first format.

        Returns:
            1D float32 feature vector.
        """
        va, vb, vc = window[CH["Va"]], window[CH["Vb"]], window[CH["Vc"]]
        ia, ib, ic = window[CH["Ia"]], window[CH["Ib"]], window[CH["Ic"]]
        freq_w = window[CH["Freq"]]
        pf_w = window[CH["PF"]]
        thd_w = window[CH["THD_V"]]
        in_w = window[CH["In"]]

        feats: Dict[str, float] = {}

        # Statistical features for all channels
        for ch_name, ch_data in [
            ("Va", va), ("Vb", vb), ("Vc", vc),
            ("Ia", ia), ("Ib", ib), ("Ic", ic),
            ("Freq", freq_w), ("PF", pf_w), ("THD_V", thd_w), ("In", in_w),
        ]:
            feats.update(self._stat_features(ch_data, ch_name))

        # FFT features for voltage and current channels
        for ch_name, ch_data in [
            ("Va", va), ("Vb", vb), ("Vc", vc),
            ("Ia", ia), ("Ib", ib), ("Ic", ic),
        ]:
            feats.update(self._fft_features(ch_data, ch_name))

        # Symmetrical components (Fortescue transform)
        feats.update(self._symmetrical_component_features(va, vb, vc, ia, ib, ic))

        # Power and imbalance features
        feats.update(self._power_features(va, vb, vc, ia, ib, ic))

        # ROCOF
        feats["rocof_rms"] = self._rocof(freq_w)

        # Cross-channel correlation and phase angle
        feats.update(self._cross_channel_features(va, vb, vc, ia, ib, ic))

        # Neutral current statistics already covered by stat_features("In")
        # Add peak neutral current as dedicated feature
        feats["In_peak"] = float(np.max(np.abs(in_w)))

        return np.array(list(feats.values()), dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names (requires one extract_window call)."""
        if self._feature_names is None:
            dummy = np.zeros((10, 256), dtype=np.float32)
            feats = self._collect_feature_dict(dummy)
            self._feature_names = list(feats.keys())
        return self._feature_names

    def _collect_feature_dict(self, window: np.ndarray) -> Dict[str, float]:
        """Same as extract_window but returns the dict (for name extraction)."""
        va, vb, vc = window[0], window[1], window[2]
        ia, ib, ic = window[3], window[4], window[5]
        freq_w = window[6]
        pf_w = window[7]
        thd_w = window[8]
        in_w = window[9]

        feats: Dict[str, float] = {}
        for ch_name, ch_data in [
            ("Va", va), ("Vb", vb), ("Vc", vc),
            ("Ia", ia), ("Ib", ib), ("Ic", ic),
            ("Freq", freq_w), ("PF", pf_w), ("THD_V", thd_w), ("In", in_w),
        ]:
            feats.update(self._stat_features(ch_data, ch_name))
        for ch_name, ch_data in [
            ("Va", va), ("Vb", vb), ("Vc", vc),
            ("Ia", ia), ("Ib", ib), ("Ic", ic),
        ]:
            feats.update(self._fft_features(ch_data, ch_name))
        feats.update(self._symmetrical_component_features(va, vb, vc, ia, ib, ic))
        feats.update(self._power_features(va, vb, vc, ia, ib, ic))
        feats["rocof_rms"] = self._rocof(freq_w)
        feats.update(self._cross_channel_features(va, vb, vc, ia, ib, ic))
        feats["In_peak"] = float(np.max(np.abs(in_w)))
        return feats

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract features for a batch of windows.

        Args:
            windows: (n_windows, n_channels, window_size) array.

        Returns:
            feature_matrix: (n_windows, n_features) float32 array.
        """
        n_windows = windows.shape[0]
        logger.info("Extracting features from %d windows...", n_windows)

        rows = []
        for i in range(n_windows):
            rows.append(self.extract_window(windows[i]))
            if (i + 1) % 5000 == 0:
                logger.debug("Processed %d / %d windows", i + 1, n_windows)

        X = np.stack(rows, axis=0).astype(np.float32)
        self._feature_names = list(self._collect_feature_dict(windows[0]).keys())
        logger.info("Feature matrix shape: %s, feature count: %d", X.shape, X.shape[1])
        return X

    def to_dataframe(self, feature_matrix: np.ndarray) -> pd.DataFrame:
        """Wrap feature matrix in a DataFrame with named columns."""
        names = self.get_feature_names()
        return pd.DataFrame(feature_matrix, columns=names)


# ---------------------------------------------------------------------------
# CLI validation gate
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature extractor validation gate")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--test", action="store_true", help="Run validation gate")
    args = parser.parse_args()

    cfg = get_config(args.config)
    setup_logging(cfg["general"]["log_level"])

    feat_cfg = cfg["features"]
    extractor = FeatureExtractor(cfg)

    logger.info("Running feature extractor validation gate...")

    # Create synthetic windows
    n_windows = 200
    n_ch = 10
    win_size = cfg["preprocessing"]["window_size"]
    rng = np.random.default_rng(42)
    dummy_windows = rng.standard_normal((n_windows, n_ch, win_size)).astype(np.float32)

    X = extractor.transform(dummy_windows)
    names = extractor.get_feature_names()

    passed = True

    # Gate 1: Feature count
    n_feats = X.shape[1]
    min_f = feat_cfg["target_feature_count_min"]
    max_f = feat_cfg["target_feature_count_max"]
    if min_f <= n_feats <= max_f:
        logger.info("Feature count: %d (in range [%d, %d]) ✓", n_feats, min_f, max_f)
    else:
        logger.error("Feature count %d outside expected range [%d, %d]", n_feats, min_f, max_f)
        passed = False

    # Gate 2: No NaN
    if not np.isnan(X).any():
        logger.info("No NaN in feature matrix ✓")
    else:
        logger.error("NaN detected in feature matrix")
        passed = False

    # Gate 3: No Inf
    if not np.isinf(X).any():
        logger.info("No Inf in feature matrix ✓")
    else:
        logger.error("Inf detected in feature matrix")
        passed = False

    print(f"\nFeature count: {n_feats}")
    print(f"Feature names (first 10): {names[:10]}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Validation {'PASSED' if passed else 'FAILED'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
