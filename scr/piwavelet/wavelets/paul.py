from __future__ import annotations

import math

import numpy as np

from .base import ArrayLike, Wavelet


class Paul(Wavelet):
    """
    Complex Paul wavelet.

    Parameters
    ----------
    m : int, default=4
        Wavelet order.
    """

    name = "Paul"

    def __init__(self, m: int = 4) -> None:
        self.m = int(m)

        if self.m < 1:
            raise ValueError("m must be >= 1")

        self.dofmin = 2

        if self.m == 4:
            self.cdelta = 1.132
            self.gamma = 1.17
            self.deltaj0 = 1.50
        else:
            self.cdelta = -1.0
            self.gamma = -1.0
            self.deltaj0 = -1.0

    def psi_ft(self, w: ArrayLike) -> ArrayLike:
        """
        Fourier-domain Paul wavelet.
        """
        norm = (
            2**self.m
            / np.sqrt(
                self.m
                * np.prod(np.arange(2, 2 * self.m))
            )
        )

        return (
            norm
            * (w**self.m)
            * np.exp(-w)
            * (w > 0)
        )

    def psi(self, t: ArrayLike) -> ArrayLike:
        """
        Time-domain Paul wavelet.
        """
        norm = (
            (2**self.m)
            * (1j**self.m)
            * math.factorial(self.m - 1)
            / np.sqrt(
                np.pi
                * math.factorial(2 * self.m)
            )
        )

        return norm * (1 - 1j * t) ** (-(self.m + 1))

    def flambda(self) -> float:
        """
        Fourier wavelength.
        """
        return (4 * np.pi) / (2 * self.m + 1)

    def coi(self) -> float:
        """
        Cone of influence size.
        """
        return np.sqrt(2.0)
