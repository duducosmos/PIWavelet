from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from piwavelet.statistics import ar1_spectrum


def test_ar1_spectrum_is_positive() -> None:
    freqs = np.linspace(0.0, 0.5, 512)

    spectrum = ar1_spectrum(freqs, alpha=0.7)

    assert np.all(spectrum > 0)


def test_ar1_spectrum_white_noise_is_flat() -> None:
    freqs = np.linspace(0.0, 0.5, 512)

    spectrum = ar1_spectrum(freqs, alpha=0.0)

    assert_allclose(
        spectrum,
        np.ones_like(freqs),
    )


def test_ar1_red_noise_has_low_frequency_power() -> None:
    freqs = np.linspace(0.0, 0.5, 512)

    spectrum = ar1_spectrum(freqs, alpha=0.8)

    assert spectrum[0] > spectrum[-1]


def test_ar1_spectrum_is_real() -> None:
    freqs = np.linspace(0.0, 0.5, 512)

    spectrum = ar1_spectrum(freqs, alpha=0.5)

    assert np.isrealobj(spectrum)
