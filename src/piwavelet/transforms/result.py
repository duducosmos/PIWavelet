from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from piwavelet.wavelets.base import BaseWavelet


@dataclass(slots=True)
class CWTResult:
    """
    Continuous Wavelet Transform result.

    Stores both the original-domain information and the
    spectral-domain representation required for downstream
    analyses such as XWT, WTC, significance testing,
    inverse transforms, and smoothing diagnostics.
    """

    # ------------------------------------------------------------------
    # original domain
    # ------------------------------------------------------------------

    signal: NDArray[np.float64]
    time: NDArray[np.float64] | None

    # ------------------------------------------------------------------
    # transform coefficients
    # shape = (n_scales, n_samples)
    # ------------------------------------------------------------------

    coefficients: NDArray[np.complex128]

    # ------------------------------------------------------------------
    # scale / frequency domain
    # ------------------------------------------------------------------

    scales: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    periods: NDArray[np.float64]

    # ------------------------------------------------------------------
    # fft domain
    # ------------------------------------------------------------------

    fft: NDArray[np.complex128]

    # angular frequencies (rad / unit time)
    angular_frequencies: NDArray[np.float64]

    # ------------------------------------------------------------------
    # edge diagnostics
    # ------------------------------------------------------------------

    coi: NDArray[np.float64]

    # ------------------------------------------------------------------
    # transform configuration
    # ------------------------------------------------------------------

    dt: float
    dj: float
    s0: float

    n_original: int
    n_padded: int

    wavelet: BaseWavelet

    # ------------------------------------------------------------------
    # derived helpers
    # ------------------------------------------------------------------

    @property
    def fft_frequencies(self) -> NDArray[np.float64]:
        """
        Fourier frequencies in cycles per unit time.
        """
        return self.angular_frequencies / (2.0 * np.pi)

    @property
    def positive_fft(self) -> NDArray[np.complex128]:
        """
        Positive-frequency FFT spectrum excluding zero frequency.
        """
        return self.fft[1: self.n_padded // 2]

    @property
    def positive_fft_frequencies(self) -> NDArray[np.float64]:
        """
        Positive Fourier frequencies excluding zero frequency.
        """
        return self.fft_frequencies[1: self.n_padded // 2]


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
