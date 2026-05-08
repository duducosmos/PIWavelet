from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AR1Result:
    """
    Result of AR(1) parameter estimation.
    """

    alpha: float
    noise_variance: float
    mu2: float


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """
    Result of wavelet significance testing.
    """

    signif: np.ndarray
    fft_theor: np.ndarray
