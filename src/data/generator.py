"""
Synthetic power system fault dataset generator.

Simulates 7 fault classes at 1 kHz sample rate with realistic temporal structure:
pre-fault normal operation → fault onset → sustained fault → post-fault.

Usage:
    python -m src.data.generator --validate
    python -m src.data.generator --output data/synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running as __main__ from project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_config, setup_logging

logger = logging.getLogger(__name__)

# Fault class constants
FAULT_CLASSES = {
    0: "Normal",
    1: "SLG",           # Single Line-to-Ground
    2: "LL",            # Line-to-Line
    3: "3PH",           # Three-Phase
    4: "Overload",
    5: "VoltageSag",
    6: "Harmonic",
}

CHANNEL_NAMES = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Freq", "PF", "THD_V", "In"]


class FaultSimulator:
    """
    Generates synthetic time-series for each power system fault class.

    Each method returns a 2D array of shape (n_samples, n_channels) where
    channels are: Va, Vb, Vc, Ia, Ib, Ic, Freq, PF, THD_V, In (neutral current).

    The temporal structure for fault samples is:
        [pre_fault_samples | fault_onset | fault_sustained | post_fault_samples]
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        gen = cfg["data_generation"]
        self.fs = gen["sample_rate"]                    # 1000 Hz
        self.v_nom = gen["voltage_nominal"]             # 220 V
        self.f_nom = gen["frequency_nominal"]           # 50 Hz
        self.noise_std_pct = gen["noise_std_pct"] / 100.0

        self.pre_fault_samples = int(gen["pre_fault_ms"] / 1000 * self.fs)   # 500
        self.post_fault_samples = int(gen["post_fault_ms"] / 1000 * self.fs) # 200

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _time(self, n: int) -> np.ndarray:
        return np.arange(n) / self.fs

    def _phase_voltages_normal(self, n: int, v_scale: float = 1.0) -> np.ndarray:
        """Three-phase voltages in normal sinusoidal form with noise."""
        t = self._time(n)
        angles = [0, -2 * np.pi / 3, 2 * np.pi / 3]
        v = np.stack(
            [self.v_nom * v_scale * np.sin(2 * np.pi * self.f_nom * t + a) for a in angles],
            axis=1,
        )  # (n, 3)
        noise = self.rng.normal(0, self.noise_std_pct * self.v_nom, v.shape)
        return v + noise

    def _phase_currents_normal(self, n: int, i_scale: float = 1.0) -> np.ndarray:
        """Three-phase currents, 30° lagging (typical inductive load), with noise."""
        t = self._time(n)
        i_nom = 50.0  # A — mid-range nominal
        angles = [-np.pi / 6, -np.pi / 6 - 2 * np.pi / 3, -np.pi / 6 + 2 * np.pi / 3]
        i = np.stack(
            [i_nom * i_scale * np.sin(2 * np.pi * self.f_nom * t + a) for a in angles],
            axis=1,
        )
        noise = self.rng.normal(0, self.noise_std_pct * i_nom, i.shape)
        return i + noise

    def _add_noise(self, arr: np.ndarray, std: float) -> np.ndarray:
        return arr + self.rng.normal(0, std, arr.shape)

    def _rms(self, arr: np.ndarray) -> np.ndarray:
        """Per-column RMS."""
        return np.sqrt(np.mean(arr ** 2, axis=0))

    def _build_sample(
        self,
        v: np.ndarray,      # (n, 3)
        i: np.ndarray,      # (n, 3)
        freq: np.ndarray,   # (n,)
        pf: np.ndarray,     # (n,)
        thd: np.ndarray,    # (n,)
        i_n: np.ndarray,    # (n,) neutral current
    ) -> np.ndarray:
        """Concatenate channels into (n, 10) array."""
        return np.column_stack([v, i, freq, pf, thd, i_n])

    # ------------------------------------------------------------------
    # Fault generators
    # ------------------------------------------------------------------

    def normal(self, n: int) -> np.ndarray:
        """Class 0: All signals in nominal range with Gaussian noise."""
        gen = self.cfg["data_generation"]
        v = self._phase_voltages_normal(n)
        i = self._phase_currents_normal(n)
        freq = self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], n)
        pf = self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], n)
        thd = self.rng.uniform(0.5, gen["thd_normal_max"], n)
        i_n = self.rng.uniform(0, gen["neutral_current_max"], n)
        return self._build_sample(v, i, freq, pf, thd, i_n)

    def single_line_to_ground(self, n: int) -> np.ndarray:
        """
        Class 1: SLG fault on phase A.
        - Phase A voltage drops 30–60%
        - Phase A current spikes 2–4×
        - Neutral current rises sharply
        - Precursor: rising harmonic distortion in pre-fault window
        """
        gen = self.cfg["data_generation"]
        pre = self.pre_fault_samples
        post = self.post_fault_samples
        fault_n = n - pre - post

        # Pre-fault: normal with rising THD precursor
        v_pre = self._phase_voltages_normal(pre)
        i_pre = self._phase_currents_normal(pre)
        thd_pre = np.linspace(2.0, gen["thd_normal_max"], pre) + self.rng.normal(0, 0.3, pre)
        thd_pre = np.clip(thd_pre, 0, gen["thd_normal_max"])
        freq_pre = self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], pre)
        pf_pre = self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], pre)
        i_n_pre = self.rng.uniform(0, gen["neutral_current_max"], pre)

        # Fault: phase A voltage collapse, current spike
        v_drop = self.rng.uniform(gen["slg_voltage_drop_min"], gen["slg_voltage_drop_max"])
        i_spike = self.rng.uniform(gen["slg_current_spike_min"], gen["slg_current_spike_max"])

        t_fault = self._time(fault_n)
        angles_v = [0, -2 * np.pi / 3, 2 * np.pi / 3]
        va_fault = self.v_nom * (1 - v_drop) * np.sin(2 * np.pi * self.f_nom * t_fault)
        vb_fault = self.v_nom * np.sin(2 * np.pi * self.f_nom * t_fault + angles_v[1])
        vc_fault = self.v_nom * np.sin(2 * np.pi * self.f_nom * t_fault + angles_v[2])
        v_fault = np.column_stack([va_fault, vb_fault, vc_fault])
        v_fault += self.rng.normal(0, self.noise_std_pct * self.v_nom, v_fault.shape)

        angles_i = [-np.pi / 6, -np.pi / 6 - 2 * np.pi / 3, -np.pi / 6 + 2 * np.pi / 3]
        i_nom = 50.0
        ia_fault = i_nom * i_spike * np.sin(2 * np.pi * self.f_nom * t_fault + angles_i[0])
        ib_fault = i_nom * np.sin(2 * np.pi * self.f_nom * t_fault + angles_i[1])
        ic_fault = i_nom * np.sin(2 * np.pi * self.f_nom * t_fault + angles_i[2])
        i_fault = np.column_stack([ia_fault, ib_fault, ic_fault])
        i_fault += self.rng.normal(0, self.noise_std_pct * i_nom, i_fault.shape)

        i_n_fault = (i_spike - 1) * i_nom * np.abs(np.sin(2 * np.pi * self.f_nom * t_fault))
        i_n_fault += self.rng.normal(0, 0.5, fault_n)

        thd_fault = self.rng.uniform(3.0, 8.0, fault_n)
        freq_fault = self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], fault_n)
        pf_fault = self.rng.uniform(0.7, 0.85, fault_n)

        # Post-fault: recovery
        v_post = self._phase_voltages_normal(post)
        i_post = self._phase_currents_normal(post)
        thd_post = self.rng.uniform(1.0, gen["thd_normal_max"], post)
        freq_post = self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], post)
        pf_post = self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], post)
        i_n_post = self.rng.uniform(0, gen["neutral_current_max"], post)

        v = np.vstack([v_pre, v_fault, v_post])
        i = np.vstack([i_pre, i_fault, i_post])
        freq = np.concatenate([freq_pre, freq_fault, freq_post])
        pf = np.concatenate([pf_pre, pf_fault, pf_post])
        thd = np.concatenate([thd_pre, thd_fault, thd_post])
        i_n = np.concatenate([i_n_pre, i_n_fault, i_n_post])
        return self._build_sample(v, i, freq, pf, thd, i_n)

    def line_to_line(self, n: int) -> np.ndarray:
        """
        Class 2: LL fault between phases A and B.
        - Phases A and B voltage dip 20–40%
        - Current imbalance between phases
        - Frequency deviation ±0.1 Hz
        """
        gen = self.cfg["data_generation"]
        pre = self.pre_fault_samples
        post = self.post_fault_samples
        fault_n = n - pre - post

        v_drop = self.rng.uniform(gen["ll_voltage_drop_min"], gen["ll_voltage_drop_max"])
        t_fault = self._time(fault_n)
        angles_v = [0, -2 * np.pi / 3, 2 * np.pi / 3]

        va_f = self.v_nom * (1 - v_drop) * np.sin(2 * np.pi * self.f_nom * t_fault + angles_v[0])
        vb_f = self.v_nom * (1 - v_drop) * np.sin(2 * np.pi * self.f_nom * t_fault + angles_v[1])
        vc_f = self.v_nom * np.sin(2 * np.pi * self.f_nom * t_fault + angles_v[2])
        v_fault = np.column_stack([va_f, vb_f, vc_f])
        v_fault += self.rng.normal(0, self.noise_std_pct * self.v_nom, v_fault.shape)

        i_nom = 50.0
        imbalance = 1.5
        angles_i = [-np.pi / 6, -np.pi / 6 - 2 * np.pi / 3, -np.pi / 6 + 2 * np.pi / 3]
        ia_f = i_nom * imbalance * np.sin(2 * np.pi * self.f_nom * t_fault + angles_i[0])
        ib_f = i_nom * imbalance * np.sin(2 * np.pi * self.f_nom * t_fault + angles_i[1])
        ic_f = i_nom * 0.7 * np.sin(2 * np.pi * self.f_nom * t_fault + angles_i[2])
        i_fault = np.column_stack([ia_f, ib_f, ic_f])
        i_fault += self.rng.normal(0, self.noise_std_pct * i_nom, i_fault.shape)

        freq_dev = self.rng.uniform(-0.1, 0.1)
        freq_fault = np.full(fault_n, self.f_nom + freq_dev) + self.rng.normal(0, 0.02, fault_n)
        thd_fault = self.rng.uniform(2.0, 6.0, fault_n)
        pf_fault = self.rng.uniform(0.75, 0.90, fault_n)
        i_n_fault = np.abs(ia_f + ib_f + ic_f) * 0.1 + self.rng.uniform(0, 1, fault_n)

        v = np.vstack([self._phase_voltages_normal(pre), v_fault, self._phase_voltages_normal(post)])
        i_arr = np.vstack([self._phase_currents_normal(pre), i_fault, self._phase_currents_normal(post)])
        freq = np.concatenate([
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], pre),
            freq_fault,
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], post),
        ])
        pf = np.concatenate([
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], pre),
            pf_fault,
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], post),
        ])
        thd = np.concatenate([
            self.rng.uniform(0.5, gen["thd_normal_max"], pre),
            thd_fault,
            self.rng.uniform(0.5, gen["thd_normal_max"], post),
        ])
        i_n = np.concatenate([
            self.rng.uniform(0, gen["neutral_current_max"], pre),
            i_n_fault,
            self.rng.uniform(0, gen["neutral_current_max"], post),
        ])
        return self._build_sample(v, i_arr, freq, pf, thd, i_n)

    def three_phase_fault(self, n: int) -> np.ndarray:
        """
        Class 3: Bolted three-phase fault.
        - All phase voltages collapse >80%
        - All phase currents spike 5–10×
        - Frequency deviation ±0.3 Hz
        """
        gen = self.cfg["data_generation"]
        pre = self.pre_fault_samples
        post = self.post_fault_samples
        fault_n = n - pre - post

        v_drop = gen["three_phase_voltage_drop"]
        i_spike = self.rng.uniform(gen["three_phase_current_spike_min"], gen["three_phase_current_spike_max"])
        freq_dev = self.rng.uniform(-0.3, 0.3)

        t_fault = self._time(fault_n)
        angles_v = [0, -2 * np.pi / 3, 2 * np.pi / 3]
        angles_i = [-np.pi / 6, -np.pi / 6 - 2 * np.pi / 3, -np.pi / 6 + 2 * np.pi / 3]
        i_nom = 50.0

        v_fault = np.column_stack([
            self.v_nom * (1 - v_drop) * np.sin(2 * np.pi * self.f_nom * t_fault + a)
            for a in angles_v
        ]) + self.rng.normal(0, self.noise_std_pct * self.v_nom, (fault_n, 3))

        i_fault = np.column_stack([
            i_nom * i_spike * np.sin(2 * np.pi * self.f_nom * t_fault + a)
            for a in angles_i
        ]) + self.rng.normal(0, self.noise_std_pct * i_nom, (fault_n, 3))

        freq_fault = np.full(fault_n, self.f_nom + freq_dev) + self.rng.normal(0, 0.05, fault_n)
        thd_fault = self.rng.uniform(4.0, 10.0, fault_n)
        pf_fault = self.rng.uniform(0.5, 0.75, fault_n)
        i_n_fault = self.rng.uniform(0.5, 3.0, fault_n)  # small due to symmetry

        v = np.vstack([self._phase_voltages_normal(pre), v_fault, self._phase_voltages_normal(post)])
        i_arr = np.vstack([self._phase_currents_normal(pre), i_fault, self._phase_currents_normal(post)])
        freq = np.concatenate([
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], pre),
            freq_fault,
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], post),
        ])
        pf = np.concatenate([
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], pre),
            pf_fault,
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], post),
        ])
        thd = np.concatenate([
            self.rng.uniform(0.5, gen["thd_normal_max"], pre),
            thd_fault,
            self.rng.uniform(0.5, gen["thd_normal_max"], post),
        ])
        i_n = np.concatenate([
            self.rng.uniform(0, gen["neutral_current_max"], pre),
            i_n_fault,
            self.rng.uniform(0, gen["neutral_current_max"], post),
        ])
        return self._build_sample(v, i_arr, freq, pf, thd, i_n)

    def overload(self, n: int) -> np.ndarray:
        """
        Class 4: Gradual overload.
        - All phase currents rise gradually over 2–10 seconds
        - Power factor degrades
        - Slight frequency dip
        """
        gen = self.cfg["data_generation"]
        pre = self.pre_fault_samples
        post = self.post_fault_samples
        fault_n = n - pre - post

        rise_start = gen["overload_current_rise_min"]
        rise_end = gen["overload_current_rise_max"]
        i_scale_ramp = np.linspace(rise_start, rise_end, fault_n)

        t_fault = self._time(fault_n)
        i_nom = 50.0
        angles_i = [-np.pi / 6, -np.pi / 6 - 2 * np.pi / 3, -np.pi / 6 + 2 * np.pi / 3]
        i_fault = np.column_stack([
            i_nom * i_scale_ramp * np.sin(2 * np.pi * self.f_nom * t_fault + a)
            for a in angles_i
        ]) + self.rng.normal(0, self.noise_std_pct * i_nom, (fault_n, 3))

        v_fault = self._phase_voltages_normal(fault_n, v_scale=0.97)  # slight voltage depression
        pf_ramp = np.linspace(gen["power_factor_max"], 0.80, fault_n) + self.rng.normal(0, 0.01, fault_n)
        pf_fault = np.clip(pf_ramp, 0.75, 1.0)
        freq_fault = np.linspace(self.f_nom, self.f_nom - 0.15, fault_n) + self.rng.normal(0, 0.01, fault_n)
        thd_fault = self.rng.uniform(2.0, 5.0, fault_n)
        i_n_fault = self.rng.uniform(0.5, 2.0, fault_n)

        v = np.vstack([self._phase_voltages_normal(pre), v_fault, self._phase_voltages_normal(post)])
        i_arr = np.vstack([self._phase_currents_normal(pre), i_fault, self._phase_currents_normal(post)])
        freq = np.concatenate([
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], pre),
            freq_fault,
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], post),
        ])
        pf = np.concatenate([
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], pre),
            pf_fault,
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], post),
        ])
        thd = np.concatenate([
            self.rng.uniform(0.5, gen["thd_normal_max"], pre),
            thd_fault,
            self.rng.uniform(0.5, gen["thd_normal_max"], post),
        ])
        i_n = np.concatenate([
            self.rng.uniform(0, gen["neutral_current_max"], pre),
            i_n_fault,
            self.rng.uniform(0, gen["neutral_current_max"], post),
        ])
        return self._build_sample(v, i_arr, freq, pf, thd, i_n)

    def voltage_sag(self, n: int) -> np.ndarray:
        """
        Class 5: Voltage sag — momentary dip due to motor starting or remote fault.
        - Phase voltages drop 10–30% for a defined duration
        """
        gen = self.cfg["data_generation"]
        pre = self.pre_fault_samples
        post = self.post_fault_samples
        fault_n = n - pre - post

        sag_depth = self.rng.uniform(gen["sag_voltage_drop_min"], gen["sag_voltage_drop_max"])
        t_fault = self._time(fault_n)
        angles_v = [0, -2 * np.pi / 3, 2 * np.pi / 3]

        v_fault = np.column_stack([
            self.v_nom * (1 - sag_depth) * np.sin(2 * np.pi * self.f_nom * t_fault + a)
            for a in angles_v
        ]) + self.rng.normal(0, self.noise_std_pct * self.v_nom, (fault_n, 3))

        i_fault = self._phase_currents_normal(fault_n, i_scale=1.1)  # slight compensatory current rise
        freq_fault = self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], fault_n)
        thd_fault = self.rng.uniform(1.5, 5.0, fault_n)
        pf_fault = self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], fault_n)
        i_n_fault = self.rng.uniform(0, 2.5, fault_n)

        v = np.vstack([self._phase_voltages_normal(pre), v_fault, self._phase_voltages_normal(post)])
        i_arr = np.vstack([self._phase_currents_normal(pre), i_fault, self._phase_currents_normal(post)])
        freq = np.concatenate([
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], pre),
            freq_fault,
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], post),
        ])
        pf = np.concatenate([
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], pre),
            pf_fault,
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], post),
        ])
        thd = np.concatenate([
            self.rng.uniform(0.5, gen["thd_normal_max"], pre),
            thd_fault,
            self.rng.uniform(0.5, gen["thd_normal_max"], post),
        ])
        i_n = np.concatenate([
            self.rng.uniform(0, gen["neutral_current_max"], pre),
            i_n_fault,
            self.rng.uniform(0, gen["neutral_current_max"], post),
        ])
        return self._build_sample(v, i_arr, freq, pf, thd, i_n)

    def harmonic_distortion(self, n: int) -> np.ndarray:
        """
        Class 6: Harmonic distortion — THD_V rises above 8%, 3rd/5th/7th harmonics.
        Often precedes transformer heating faults.
        """
        gen = self.cfg["data_generation"]
        pre = self.pre_fault_samples
        post = self.post_fault_samples
        fault_n = n - pre - post

        t_fault = self._time(fault_n)
        angles_v = [0, -2 * np.pi / 3, 2 * np.pi / 3]

        # Fundamental + 3rd + 5th + 7th harmonics
        h3_amp = self.rng.uniform(0.05, 0.12)   # 5–12% of fundamental
        h5_amp = self.rng.uniform(0.03, 0.08)
        h7_amp = self.rng.uniform(0.02, 0.05)

        v_cols = []
        for a in angles_v:
            v_fund = self.v_nom * np.sin(2 * np.pi * self.f_nom * t_fault + a)
            v_h3 = self.v_nom * h3_amp * np.sin(2 * np.pi * 3 * self.f_nom * t_fault + 3 * a)
            v_h5 = self.v_nom * h5_amp * np.sin(2 * np.pi * 5 * self.f_nom * t_fault + 5 * a)
            v_h7 = self.v_nom * h7_amp * np.sin(2 * np.pi * 7 * self.f_nom * t_fault + 7 * a)
            v_cols.append(v_fund + v_h3 + v_h5 + v_h7)

        v_fault = np.column_stack(v_cols)
        v_fault += self.rng.normal(0, self.noise_std_pct * self.v_nom, v_fault.shape)

        i_fault = self._phase_currents_normal(fault_n, i_scale=1.05)
        thd_computed = np.sqrt(h3_amp ** 2 + h5_amp ** 2 + h7_amp ** 2) * 100
        thd_fault = np.full(fault_n, thd_computed) + self.rng.normal(0, 0.5, fault_n)
        thd_fault = np.clip(thd_fault, gen["harmonic_thd_threshold"], 25.0)
        freq_fault = self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], fault_n)
        pf_fault = self.rng.uniform(0.80, 0.92, fault_n)
        i_n_fault = self.rng.uniform(0.5, 3.5, fault_n)  # triplen harmonics cause neutral current

        v = np.vstack([self._phase_voltages_normal(pre), v_fault, self._phase_voltages_normal(post)])
        i_arr = np.vstack([self._phase_currents_normal(pre), i_fault, self._phase_currents_normal(post)])
        freq = np.concatenate([
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], pre),
            freq_fault,
            self.rng.uniform(gen["frequency_normal_min"], gen["frequency_normal_max"], post),
        ])
        pf = np.concatenate([
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], pre),
            pf_fault,
            self.rng.uniform(gen["power_factor_min"], gen["power_factor_max"], post),
        ])
        thd = np.concatenate([
            self.rng.uniform(0.5, gen["thd_normal_max"], pre),
            thd_fault,
            self.rng.uniform(0.5, gen["thd_normal_max"], post),
        ])
        i_n = np.concatenate([
            self.rng.uniform(0, gen["neutral_current_max"], pre),
            i_n_fault,
            self.rng.uniform(0, gen["neutral_current_max"], post),
        ])
        return self._build_sample(v, i_arr, freq, pf, thd, i_n)


class DatasetGenerator:
    """
    Orchestrates FaultSimulator to generate a balanced, stratified dataset.

    Produces:
    - Raw time-series CSV (data/synthetic/raw_timeseries.csv)
    - Split CSVs: train.csv, val.csv, test.csv
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        seed = cfg["general"]["random_seed"]
        self.rng = np.random.default_rng(seed)
        self.sim = FaultSimulator(cfg, self.rng)
        self.samples_per_class = cfg["data_generation"]["samples_per_class"]

    def _generate_class(self, fault_id: int) -> np.ndarray:
        """Generate samples_per_class rows for a single fault class."""
        n = self.samples_per_class
        fn_map = {
            0: self.sim.normal,
            1: self.sim.single_line_to_ground,
            2: self.sim.line_to_line,
            3: self.sim.three_phase_fault,
            4: self.sim.overload,
            5: self.sim.voltage_sag,
            6: self.sim.harmonic_distortion,
        }
        logger.info("Generating class %d (%s): %d samples", fault_id, FAULT_CLASSES[fault_id], n)
        data = fn_map[fault_id](n)
        return data

    def generate(self) -> pd.DataFrame:
        """Generate the full balanced dataset and return as DataFrame."""
        frames: List[pd.DataFrame] = []
        for fault_id in FAULT_CLASSES:
            data = self._generate_class(fault_id)
            df = pd.DataFrame(data, columns=CHANNEL_NAMES)
            df["label"] = fault_id
            df["fault_name"] = FAULT_CLASSES[fault_id]
            frames.append(df)

        full_df = pd.concat(frames, ignore_index=True)
        # Shuffle while maintaining reproducibility
        full_df = full_df.sample(frac=1, random_state=self.cfg["general"]["random_seed"]).reset_index(drop=True)
        logger.info("Total dataset: %d samples, %d columns", len(full_df), len(full_df.columns))
        return full_df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train/val/test split."""
        gen = self.cfg["data_generation"]
        train_r, val_r = gen["train_ratio"], gen["val_ratio"]

        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(
            df, test_size=(1 - train_r), stratify=df["label"],
            random_state=self.cfg["general"]["random_seed"]
        )
        val_size = val_r / (gen["val_ratio"] + gen["test_ratio"])
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size), stratify=temp_df["label"],
            random_state=self.cfg["general"]["random_seed"]
        )
        logger.info("Split sizes — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
        return train_df, val_df, test_df

    def save(self, df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
        """Save raw timeseries and split CSVs to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df, val_df, test_df = self.split(df)

        paths = {}
        for name, frame in [("raw_timeseries", df), ("train", train_df), ("val", val_df), ("test", test_df)]:
            p = output_dir / f"{name}.csv"
            frame.to_csv(p, index=False)
            paths[name] = p
            logger.info("Saved %s.csv (%d rows)", name, len(frame))
        return paths

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validation gate for Phase 1:
        - All 7 classes present
        - No NaN values
        - >= samples_per_class rows per class
        """
        passed = True

        missing_classes = set(FAULT_CLASSES.keys()) - set(df["label"].unique())
        if missing_classes:
            logger.error("Missing classes: %s", missing_classes)
            passed = False

        if df[CHANNEL_NAMES].isnull().any().any():
            logger.error("NaN values detected in sensor columns")
            passed = False

        class_counts = df["label"].value_counts()
        for cls_id in FAULT_CLASSES:
            cnt = class_counts.get(cls_id, 0)
            if cnt < self.samples_per_class:
                logger.error("Class %d has only %d samples (need %d)", cls_id, cnt, self.samples_per_class)
                passed = False

        if passed:
            logger.info("Validation PASSED — all gates cleared")
        return passed

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print class distribution and sample statistics."""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(df):,}")
        print(f"Channels: {CHANNEL_NAMES}")
        print()

        print("Class Distribution:")
        print("-" * 40)
        class_counts = df["label"].value_counts().sort_index()
        for cls_id, cnt in class_counts.items():
            print(f"  Class {cls_id:1d} ({FAULT_CLASSES[cls_id]:<16s}): {cnt:>8,} samples")

        print()
        print("Channel Statistics (mean ± std):")
        print("-" * 40)
        for col in CHANNEL_NAMES:
            m = df[col].mean()
            s = df[col].std()
            print(f"  {col:<8s}: {m:>10.3f} ± {s:.3f}")
        print("=" * 60)

    def save_sample_plots(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Save 3 sample plots per fault class to output_dir/plots/."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dpi = self.cfg["evaluation"]["plot_dpi"]

        for cls_id, cls_name in FAULT_CLASSES.items():
            subset = df[df["label"] == cls_id]
            for i in range(min(3, len(subset) // 1000)):
                start = i * (len(subset) // 3)
                chunk = subset.iloc[start : start + 1000]

                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                fig.suptitle(f"Class {cls_id}: {cls_name} — Sample {i+1}", fontsize=13)

                axes[0].plot(chunk["Va"].values, label="Va", alpha=0.8)
                axes[0].plot(chunk["Vb"].values, label="Vb", alpha=0.8)
                axes[0].plot(chunk["Vc"].values, label="Vc", alpha=0.8)
                axes[0].set_ylabel("Voltage (V)")
                axes[0].legend(loc="upper right", fontsize=8)

                axes[1].plot(chunk["Ia"].values, label="Ia", alpha=0.8)
                axes[1].plot(chunk["Ib"].values, label="Ib", alpha=0.8)
                axes[1].plot(chunk["Ic"].values, label="Ic", alpha=0.8)
                axes[1].set_ylabel("Current (A)")
                axes[1].legend(loc="upper right", fontsize=8)

                axes[2].plot(chunk["Freq"].values, label="Freq", color="green")
                axes[2].set_ylabel("Freq (Hz)")
                axes[2].set_xlabel("Sample index")

                plt.tight_layout()
                fname = plots_dir / f"class{cls_id}_{cls_name}_sample{i+1}.png"
                plt.savefig(fname, dpi=dpi, bbox_inches="tight")
                plt.close()

        logger.info("Sample plots saved to %s", plots_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic power fault dataset generator")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--validate", action="store_true", help="Run validation gate after generation")
    args = parser.parse_args()

    cfg = get_config(args.config)
    setup_logging(cfg["general"]["log_level"])

    output_dir = Path(args.output) if args.output else Path(cfg["paths"]["synthetic_data"])
    gen = DatasetGenerator(cfg)

    logger.info("Starting synthetic dataset generation...")
    df = gen.generate()
    gen.print_summary(df)

    paths = gen.save(df, output_dir)
    logger.info("Saved dataset files: %s", list(paths.keys()))

    gen.save_sample_plots(df, output_dir)

    if args.validate:
        ok = gen.validate(df)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
