from __future__ import annotations

from typing import Protocol


class WaveletProtocol(Protocol):
    """
    Protocol describing the minimum wavelet interface required by
    statistical significance routines.
    """

    dofmin: float
    cdelta: float
    gamma: float
    deltaj0: float

    def flambda(self) -> float:
        """
        Fourier wavelength factor.
        """
