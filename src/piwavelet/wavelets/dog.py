from __future__ import annotations

import math

import numpy as np
from scipy.special import eval_hermitenorm, gamma

from .base import ArrayLike, BaseWavelet


class DOG(BaseWavelet):
    """
    Derivative of Gaussian (DOG) wavelet.

    Parameters
    ----------
    m : int, default=2
        Derivative order.

    Notes
    -----
    m=2 corresponds to the Mexican Hat wavelet.
    """

    name = "DOG"

    def __init__(self, m: int = 2) -> None:
        self.m = int(m)

        if self.m < 1:
            raise ValueError("m must be >= 1")

        self.dofmin = 1

        if self.m == 2:
            self.cdelta = 3.541
            self.gamma = 1.43
            self.deltaj0 = 1.40

        elif self.m == 6:
            self.cdelta = 1.966
            self.gamma = 1.37
            self.deltaj0 = 0.97

        else:
            self.cdelta = -1.0
            self.gamma = -1.0
            self.deltaj0 = -1.0

    def psi_ft(self, w: ArrayLike) -> ArrayLike:
        """
        Fourier-domain DOG wavelet.
        """
        norm = (
            -(1j**self.m)
            / np.sqrt(gamma(self.m + 0.5))
        )

        return (
            norm
            * (w**self.m)
            * np.exp(-0.5 * w**2)
        )

    def psi(self, t: ArrayLike) -> ArrayLike:
        """
        Time-domain DOG wavelet.
        """
        hermite = eval_hermitenorm(self.m, t)

        norm = (
            (-1) ** (self.m + 1)
            / np.sqrt(gamma(self.m + 0.5))
        )

        return (
            norm
            * hermite
            * np.exp(-0.5 * t**2)
        )

    def flambda(self) -> float:
        """
        Fourier wavelength.
        """
        return (2 * np.pi) / np.sqrt(self.m + 0.5)

    def coi(self) -> float:
        """
        Cone of influence size.
        """
        return 1.0 / np.sqrt(2.0)

    def time_smoothing_scale(
        self,
        scale: float,
    ) -> float:
        """
        Temporal decorrelation scale for
        wavelet coherence smoothing.

        DOG wavelets are real-valued and
        strongly time-localized.
        """

        return scale * 0.6
