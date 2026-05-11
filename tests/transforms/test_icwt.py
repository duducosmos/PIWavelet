from __future__ import annotations

import numpy as np

from piwavelet.transforms.cwt import cwt
from piwavelet.transforms.icwt import icwt
from piwavelet.wavelets.morlet import Morlet


def test_icwt_reconstructs_signal():
    """
    ICWT should reconstruct the original signal
    with very small numerical error.
    """

    dt = 0.01

    t = np.arange(
        0,
        10,
        dt,
    )

    signal = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 15 * t)
    )

    wavelet = Morlet()

    coefficients, scales, freqs, coi = cwt(
        signal=signal,
        dt=dt,
        dj=1 / 12,
        wavelet=wavelet,
    )

    reconstructed = icwt(
        coefficients=coefficients,
        scales=scales,
        dt=dt,
        dj=1 / 12,
        wavelet=wavelet,
    )

    # -----------------------------------------------------
    # RMS reconstruction error
    # -----------------------------------------------------

    rms_error = np.sqrt(
        np.mean(
            (signal - reconstructed) ** 2
        )
    )

    assert rms_error < 1e-6

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    correlation = np.corrcoef(
        signal,
        reconstructed,
    )[0, 1]

    assert correlation > 0.9999
