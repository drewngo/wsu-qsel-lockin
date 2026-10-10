# Copyright (c) 2026 Wichita State University
# Author: Andrew Ngo, Quantum Science and Engineering Lab
"""Sanity checks for model/sensor_model.py: dip shape, spectrum, ADC range."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from sensor_model import (  # noqa: E402
    ADC_MAX,
    ADC_MIN,
    F_MOD,
    SensorParams,
    detector_output,
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 20_000.0  # sample rate, Hz (roughly post-decimation)

    # --- Dip test: no wiggle, sweep the offset ---
    p_static = SensorParams(f_dev=0.0)
    offsets = np.linspace(-3, 3, 601)
    v_dip, _ = detector_output(np.zeros_like(offsets), offsets, p_static)
    i_half = np.argmin(np.abs(offsets - 0.5))
    expected = p_static.v_offset - p_static.amp / 2
    print(f"Dip test:      V(+0.5) = {v_dip[i_half]:.4f} V, expected {expected:.4f} V")

    # --- Spectrum test: 1 s of data so FFT bins are 1 Hz apart ---
    p = SensorParams()
    t = np.arange(int(fs)) / fs
    freqs = np.fft.rfftfreq(len(t), 1 / fs)
    i77 = np.argmin(np.abs(freqs - F_MOD))
    i154 = np.argmin(np.abs(freqs - 2 * F_MOD))

    spectra = {}
    for off in (0.5, 0.0):
        v, codes = detector_output(t, off, p)
        spec = 2 * np.abs(np.fft.rfft(v - v.mean())) / len(t)  # single-sided amplitude
        spectra[off] = spec
        print(f"Spectrum test: offset {off}: 77 Hz = {spec[i77]:.5f} V, "
              f"154 Hz = {spec[i154]:.5f} V")

    # --- Clipping test ---
    print(f"Clipping test: codes span [{codes.min()}, {codes.max()}] "
          f"of [{ADC_MIN}, {ADC_MAX}]")

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(offsets, v_dip)
    ax1.axvline(0.5, ls="--", lw=0.8, color="gray")
    ax1.axvline(-0.5, ls="--", lw=0.8, color="gray")
    ax1.set_xlabel("Synthesizer offset (linewidths)")
    ax1.set_ylabel("Detector output (V)")
    ax1.set_title("Dip (no modulation)")

    for off, spec in spectra.items():
        ax2.plot(freqs, spec, label=f"offset = {off}")
    ax2.set_xlim(0, 400)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude (V)")
    ax2.set_title("Spectrum with 77 Hz modulation")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("sensor_model_checks.png", dpi=150)
    plt.show()