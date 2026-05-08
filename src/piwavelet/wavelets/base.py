from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


ArrayLike = NDArray[np.float64] | NDArray[np.complex128]


class BaseWavelet(ABC):
    """
    Base class for continuous wavelets.
    """

    name: str

    dofmin: int
    cdelta: float
    gamma: float
    deltaj0: float

    @abstractmethod
    def psi(self, t: ArrayLike) -> ArrayLike:
        """
        Wavelet in time domain.
        """
        raise NotImplementedError

    @abstractmethod
    def psi_ft(self, w: ArrayLike) -> ArrayLike:
        """
        Wavelet in Fourier domain.
        """
        raise NotImplementedError

    @abstractmethod
    def flambda(self) -> float:
        """
        Fourier wavelength.
        """
        raise NotImplementedError

    @abstractmethod
    def coi(self) -> float:
        """
        Cone of influence size.
        """
        raise NotImplementedError

    def support(self) -> float:
        """
        Effective wavelet support.
        """
        return 1.0 / self.coi()
