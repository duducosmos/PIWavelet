from __future__ import annotations

import numpy as np
import pytest

from piwavelet.wavelets import DOG


def test_dog_metadata_m2() -> None:
    wavelet = DOG(m=2)

    assert wavelet.dofmin == 1
    assert wavelet.cdelta == pytest.approx(3.541)
    assert wavelet.gamma == pytest.approx(1.43)
    assert wavelet.deltaj0 == pytest.approx(1.40)


def test_dog_metadata_m6() -> None:
    wavelet = DOG(m=6)

    assert wavelet.cdelta == pytest.approx(1.966)
    assert wavelet.gamma == pytest.approx(1.37)
    assert wavelet.deltaj0 == pytest.approx(0.97)


def test_dog_flambda() -> None:
    wavelet = DOG(m=2)

    expected = (2 * np.pi) / np.sqrt(2.5)

    assert wavelet.flambda() == pytest.approx(expected)


def test_dog_real_wavelet() -> None:
    wavelet = DOG(m=2)

    t = np.linspace(-5, 5, 128)

    psi = wavelet.psi(t)

    assert np.isrealobj(psi)


def test_dog_psi_shape() -> None:
    wavelet = DOG()

    t = np.linspace(-5, 5, 128)

    psi = wavelet.psi(t)

    assert psi.shape == t.shape


def test_dog_psi_ft_shape() -> None:
    wavelet = DOG()

    w = np.linspace(-10, 10, 128)

    psi_ft = wavelet.psi_ft(w)

    assert psi_ft.shape == w.shape
