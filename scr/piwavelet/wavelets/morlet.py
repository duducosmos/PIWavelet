from __future__ import annotations

import numpy as np

from .base import Wavelet


class Morlet(Wavelet):
    """
    Complex Morlet wavelet.

    Parameters
    ----------
    f0 : float, default=6.0
        Central frequency.
    """

    name = "Morlet"

    def __init__(self, f0: float = 6.0) -> None:
        self.f0 = float(f0)

        self.dofmin = 2

        if self.f0 == 6.0:
            self.cdelta = 0.776
            self.gamma = 2.32
            self.deltaj0 = 0.60
        else:
            self.cdelta = -1.0
            self.gamma = -1.0
            self.deltaj0 = -1.0

    def psi_ft(self, w):
        return (np.pi ** -0.25) * np.exp(-0.5 * (w - self.f0) ** 2)

    def psi(self, t):
        return (
            np.pi ** -0.25
            * np.exp(1j * self.f0 * t)
            * np.exp(-(t**2) / 2)
        )

    def flambda(self) -> float:
        return (4 * np.pi) / (
            self.f0 + np.sqrt(2 + self.f0**2)
        )

    def coi(self) -> float:
        return 1 / np.sqrt(2)
