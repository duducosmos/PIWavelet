from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from piwavelet.wavelets.base import BaseWavelet


def icwt(
    coefficients: NDArray[np.complex128],
    scales: NDArray[np.float64],
    dt: float,
    dj: float,
    wavelet: BaseWavelet,
) -> NDArray[np.float64]:
    """
    Inverse Continuous Wavelet Transform.

    Torrence & Compo (1998), equation (11).
    """

    coefficients = np.asarray(
        coefficients,
        dtype=np.complex128,
    )

    scales = np.asarray(
        scales,
        dtype=np.float64,
    )

    if coefficients.ndim != 2:
        raise ValueError(
            "coefficients must be 2-dimensional"
        )

    if scales.ndim != 1:
        raise ValueError(
            "scales must be one-dimensional"
        )

    if coefficients.shape[0] != scales.size:
        raise ValueError(
            "number of scales does not match "
            "coefficient matrix"
        )

    scales = scales[:, np.newaxis]

    reconstruction = (
        dj
        * np.sqrt(dt)
        / (wavelet.cdelta * wavelet.psi(0))
    )

    reconstruction *= np.sum(
        np.real(coefficients) / np.sqrt(scales),
        axis=0,
    )

    return reconstruction.astype(np.float64)
