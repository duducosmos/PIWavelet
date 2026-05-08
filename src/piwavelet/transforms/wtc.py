from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from piwavelet.smoothing.operators import (
    smooth_wavelet,
)

from piwavelet.wavelets.base import BaseWavelet

from .cwt import cwt
from .result import (
    WaveletCoherenceResult,
)


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

    Computes the magnitude-squared wavelet coherence
    following the Torrence-Webster formulation.

    Parameters
    ----------
    x : ArrayLike
        First input signal.

    y : ArrayLike
        Second input signal.

    dt : float, default=1.0
        Sampling interval.

    dj : float, default=1/12
        Scale spacing.

    s0 : float | None, default=None
        Smallest wavelet scale.

    J : int | None, default=None
        Number of scales.

    wavelet : BaseWavelet | None, default=None
        Mother wavelet.

    Returns
    -------
    WaveletCoherenceResult
        Wavelet coherence result.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "x and y must be 1D arrays"
        )

    if len(x) != len(y):
        raise ValueError(
            "x and y must have the same length"
        )

    if dt <= 0:
        raise ValueError(
            "dt must be positive"
        )

    # remove mean
    x = x - np.mean(x)
    y = y - np.mean(y)

    time = np.arange(len(x)) * dt

    # continuous wavelet transforms
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

    # cross-wavelet transform
    Wxy = (
        Wx.coefficients
        * np.conj(Wy.coefficients)
    )

    # scale normalization
    scales = Wx.scales[:, np.newaxis]

    # smoothed cross spectrum
    Sxy = smooth_wavelet(
        Wxy / scales,
        scales=Wx.scales,
        dt=dt,
        dj=dj,
        wavelet=wavelet,
    )

    # smoothed auto spectra
    Sxx = smooth_wavelet(
        (
            np.abs(Wx.coefficients) ** 2
        ) / scales,
        scales=Wx.scales,
        dt=dt,
        dj=dj,
        wavelet=wavelet,
    )

    Syy = smooth_wavelet(
        (
            np.abs(Wy.coefficients) ** 2
        ) / scales,
        scales=Wx.scales,
        dt=dt,
        dj=dj,
        wavelet=wavelet,
    )

    # numerical stability
    eps = np.finfo(
        np.float64
    ).eps

    # auto spectra should be real-positive
    Sxx = np.real(Sxx)
    Syy = np.real(Syy)

    # magnitude-squared wavelet coherence
    coherence = (
        np.abs(Sxy) ** 2
        / (
            Sxx * Syy + eps
        )
    )

    coherence = np.real(
        coherence
    )

    coherence = np.nan_to_num(
        coherence,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    coherence = np.clip(
        coherence,
        0.0,
        1.0,
    )

    # phase of smoothed cross-spectrum
    phase = np.angle(Sxy)

    return WaveletCoherenceResult(
        time=time,
        coherence=coherence,
        cross_wavelet=Wxy,
        phase=phase,
        scales=Wx.scales,
        frequencies=Wx.frequencies,
        periods=Wx.periods,
        coi=Wx.coi,
        significance=None,
    )
