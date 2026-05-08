from __future__ import annotations

import numpy as np
import pytest

from piwavelet.wavelets import Morlet


def test_morlet_metadata() -> None:
    wavelet = Morlet()

    assert wavelet.name == "Morlet"
    assert wavelet.dofmin == 2
    assert wavelet.cdelta == pytest.approx(0.776)
    assert wavelet.gamma == pytest.approx(2.32)
    assert wavelet.deltaj0 == pytest.approx(0.60)


def test_morlet_flambda() -> None:
    wavelet = Morlet(f0=6.0)

    expected = (4 * np.pi) / (6 + np.sqrt(38))

    assert wavelet.flambda() == pytest.approx(expected)


def test_morlet_coi() -> None:
    wavelet = Morlet()

    assert wavelet.coi() == pytest.approx(1 / np.sqrt(2))


def test_morlet_psi_shape() -> None:
    wavelet = Morlet()

    t = np.linspace(-5, 5, 128)

    psi = wavelet.psi(t)

    assert psi.shape == t.shape
    assert np.iscomplexobj(psi)


def test_morlet_psi_ft_shape() -> None:
    wavelet = Morlet()

    w = np.linspace(-10, 10, 128)

    psi_ft = wavelet.psi_ft(w)

    assert psi_ft.shape == w.shape


def test_morlet_peak_frequency() -> None:
    wavelet = Morlet(f0=6)

    w = np.linspace(0, 12, 10000)

    psi_ft = np.abs(wavelet.psi_ft(w))

    peak = w[np.argmax(psi_ft)]

    assert peak == pytest.approx(6.0, abs=1e-2)
