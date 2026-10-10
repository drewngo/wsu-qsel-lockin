# Copyright (c) 2026 Wichita State University
# Author: Andrew Ngo, Quantum Science and Engineering Lab
"""Synthetic detector signal for the magnetometer lock-in.

Stands in for everything upstream of the Red Pitaya ADC: modulation coil,
sensor, microwave synthesizer, detector, and analog front end.
All frequency offsets are in units of the dip width (gamma = 1).
"""

from dataclasses import dataclass
import numpy as np

F_MOD = 77.0                          # coil modulation frequency, Hz
ADC_BITS = 14
ADC_FULL_SCALE_V = 1.0                # Red Pitaya LV input range: +/-1 V
ADC_MAX = 2 ** (ADC_BITS - 1) - 1     # 8191
ADC_MIN = -(2 ** (ADC_BITS - 1))      # -8192


@dataclass
class SensorParams:
    gamma: float = 1.0        # dip width; everything else is relative to it
    f_dev: float = 0.25       # coil swing, in linewidths
    phi_coil: float = 0.0     # coil lag, radians
    amp: float = 0.3          # dip depth A, volts
    v_offset: float = 0.5     # detector baseline, volts
    noise_sigma: float = 0.0  # white noise std dev, volts
    seed: int = 0


def detector_output(t, f_offset, p=None):
    """Return (volts, adc_codes) for each sample time in t.

    t        : sample times, seconds (array)
    f_offset : synthesizer minus resonance, in linewidths (scalar or array)
    p        : SensorParams
    """
    p = p or SensorParams()
    t = np.asarray(t, dtype=float)
    f_offset = np.broadcast_to(np.asarray(f_offset, dtype=float), t.shape)

    # 1. Instantaneous offset: the coil wiggles the resonance at 77 Hz
    delta = f_offset - p.f_dev * np.sin(2 * np.pi * F_MOD * t + p.phi_coil)

    # 2. Lorentzian dip: 1 at center, 0.5 at +/- gamma/2
    dip = 1.0 / (1.0 + (2.0 * delta / p.gamma) ** 2)

    # 3. Detector voltage: baseline minus the dip, plus noise
    rng = np.random.default_rng(p.seed)
    noise = rng.normal(0.0, p.noise_sigma, t.shape)
    volts = p.v_offset - p.amp * dip + noise

    # 4. ADC: scale to codes, round, saturate, then cast
    scaled = volts / ADC_FULL_SCALE_V * 2 ** (ADC_BITS - 1)
    codes = np.clip(np.round(scaled), ADC_MIN, ADC_MAX).astype(np.int16)

    return volts, codes
