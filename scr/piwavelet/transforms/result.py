from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from piwavelet.wavelets.base import BaseWavelet


@dataclass(slots=True)
class CWTResult:
    coefficients: NDArray[np.complex128]
    scales: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    periods: NDArray[np.float64]
    coi: NDArray[np.float64]
    fft: NDArray[np.complex128]
    fft_frequencies: NDArray[np.float64]
    dt: float
    dj: float
    s0: float
    wavelet: BaseWavelet


@dataclass(slots=True)
class XWTResult:
    cross_wavelet: NDArray[np.complex128]
    power: NDArray[np.float64]
    phase: NDArray[np.float64]
    scales: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    periods: NDArray[np.float64]
    coi: NDArray[np.float64]
    xwt_significance: NDArray[np.float64] | None


@dataclass(slots=True)
class WaveletCoherenceResult:
    coherence: NDArray[np.float64]
    cross_wavelet: NDArray[np.complex128]
    phase: NDArray[np.float64]
    scales: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    periods: NDArray[np.float64]
    coi: NDArray[np.float64]
    significance: NDArray[np.float64] | None
