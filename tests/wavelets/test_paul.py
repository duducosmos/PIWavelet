from __future__ import annotations

import numpy as np
import pytest

from piwavelet.wavelets import Paul


def test_paul_metadata() -> None:
    wavelet = Paul(m=4)

    assert wavelet.dofmin == 2
    assert wavelet.cdelta == pytest.approx(1.132)
    assert wavelet.gamma == pytest.approx(1.17)
    assert wavelet.deltaj0 == pytest.approx(1.50)


def test_paul_flambda() -> None:
    wavelet = Paul(m=4)

    expected = 4 * np.pi / 9

    assert wavelet.flambda() == pytest.approx(expected)


def test_paul_positive_frequencies_only() -> None:
    wavelet = Paul()

    w = np.array([-5.0, -1.0, 0.0])

    result = wavelet.psi_ft(w)

    assert np.allclose(result, 0.0)


def test_paul_psi_shape() -> None:
    wavelet = Paul()

    t = np.linspace(-5, 5, 256)

    psi = wavelet.psi(t)

    assert psi.shape == t.shape
    assert np.iscomplexobj(psi)
