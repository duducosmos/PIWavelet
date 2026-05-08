from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from piwavelet.transforms.cwt import cwt
from piwavelet.transforms.result import XWTResult

from piwavelet.wavelets.base import BaseWavelet


def xwt(
    x: ArrayLike,
    y: ArrayLike,
    dt: float = 1.0,
    dj: float = 1 / 12,
    s0: float | None = None,
    J: int | None = None,
    wavelet: BaseWavelet | None = None,
) -> XWTResult:
    """
    Cross Wavelet Transform.
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

    power = np.abs(Wxy)

    phase = np.angle(Wxy)

    return XWTResult(
        cross_wavelet=Wxy,
        power=power,
        phase=phase,
        scales=Wx.scales,
        frequencies=Wx.frequencies,
        periods=Wx.periods,
        coi=Wx.coi,
        xwt_significance=None,
    )
