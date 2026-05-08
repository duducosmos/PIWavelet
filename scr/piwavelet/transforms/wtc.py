from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from piwavelet.smoothing.torrence_webster import (
    smooth_wavelet,
)

from piwavelet.transforms.cwt import cwt
from piwavelet.transforms.result import (
    WaveletCoherenceResult,
)

from piwavelet.wavelets.base import BaseWavelet


def wavelet_coherence(
    x: ArrayLike,
    y: ArrayLike,
    dt: float = 1.0,
    dj: float = 1 / 12,
    s0: float | None = None,
    J: int | None = None,
    wavelet: BaseWavelet | None = None,
) -> WaveletCoherenceResult:
    """
    Wavelet Coherence Transform.
    """

    Wx = cwt(
        signal=x,
        dt=dt,
        dj=dj,
        s0=s0,
        J=J,
        wavelet=wavelet,
    )

    Wy = cwt(
        signal=y,
        dt=dt,
        dj=dj,
        s0=s0,
        J=J,
        wavelet=wavelet,
    )

    Wxy = (
        Wx.coefficients
        * np.conj(Wy.coefficients)
    )

    scales = Wx.scales[:, np.newaxis]

    Sxy = smooth_wavelet(
        Wxy / scales,
        dt=dt,
        dj=dj,
        scales=Wx.scales,
        wavelet=wavelet,
    )

    Sxx = smooth_wavelet(
        np.abs(Wx.coefficients) ** 2 / scales,
        dt=dt,
        dj=dj,
        scales=Wx.scales,
        wavelet=wavelet,
    )

    Syy = smooth_wavelet(
        np.abs(Wy.coefficients) ** 2 / scales,
        dt=dt,
        dj=dj,
        scales=Wx.scales,
        wavelet=wavelet,
    )

    coherence = (
        np.abs(Sxy) ** 2
        / (Sxx * Syy)
    )

    coherence = np.clip(coherence, 0.0, 1.0)

    phase = np.angle(Wxy)

    return WaveletCoherenceResult(
        coherence=coherence,
        cross_wavelet=Wxy,
        phase=phase,
        scales=Wx.scales,
        frequencies=Wx.frequencies,
        periods=Wx.periods,
        coi=Wx.coi,
        significance=None,
    )
