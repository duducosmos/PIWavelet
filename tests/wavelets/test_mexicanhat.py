from __future__ import annotations

import numpy as np

from piwavelet.wavelets import DOG, MexicanHat


def test_mexican_hat_is_dog_order_2() -> None:
    mexican = MexicanHat()
    dog = DOG(m=2)

    t = np.linspace(-5, 5, 512)

    assert np.allclose(
        mexican.psi(t),
        dog.psi(t),
    )


def test_mexican_hat_metadata() -> None:
    wavelet = MexicanHat()

    assert wavelet.dofmin == 1
    assert wavelet.cdelta == 3.541
