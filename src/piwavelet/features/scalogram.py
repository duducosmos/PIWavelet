from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from piwavelet.transforms import cwt
from piwavelet.wavelets.base import BaseWavelet

from .normalization import (
    minmax_normalize,
    standardize,
)


def wavelet_scalogram(
    signal: NDArray[np.float64],
    *,
    dt: float = 1.0,
    dj: float = 1 / 12,
    s0: float | None = None,
    J: int | None = None,
    wavelet: BaseWavelet | None = None,
    output: str = "power",
    log_transform: bool = True,
    normalization: str | None = "minmax",
    dtype: np.dtype = np.float32,
) -> NDArray:
    """
    Convert a 1D signal into a 2D wavelet representation.

    Parameters
    ----------
    signal
        Input signal.

    dt
        Sampling interval.

    dj
        Scale spacing.

    s0
        Smallest scale.

    J
        Number of scales minus one.

    wavelet
        Mother wavelet.

    output
        Representation type.

        Options:

            - "power"
            - "amplitude"
            - "complex"
            - "real"
            - "imag"
            - "phase"

    log_transform
        Apply logarithmic compression.

    normalization
        Optional normalization.

        Options:

            - None
            - "minmax"
            - "standard"

    dtype
        Output dtype.

    Returns
    -------
    np.ndarray

        Shape:

            (n_scales, n_time)
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    result = cwt(
        signal=signal,
        dt=dt,
        dj=dj,
        s0=s0,
        J=J,
        wavelet=wavelet,
    )

    W = result.coefficients

    # ------------------------------------------------------------------
    # representation
    # ------------------------------------------------------------------

    if output == "power":

        scalogram = np.abs(W) ** 2

    elif output == "amplitude":

        scalogram = np.abs(W)

    elif output == "complex":

        scalogram = W

    elif output == "real":

        scalogram = W.real

    elif output == "imag":

        scalogram = W.imag

    elif output == "phase":

        scalogram = np.angle(W)

    else:

        raise ValueError(
            f"invalid output={output!r}"
        )

    # ------------------------------------------------------------------
    # log compression
    # ------------------------------------------------------------------

    if (
        log_transform
        and np.isrealobj(scalogram)
    ):

        scalogram = np.log1p(
            np.maximum(
                scalogram,
                0.0,
            )
        )

    # ------------------------------------------------------------------
    # normalization
    # ------------------------------------------------------------------

    if normalization is not None:

        if normalization == "minmax":

            scalogram = minmax_normalize(
                scalogram
            )

        elif normalization == "standard":

            scalogram = standardize(
                scalogram
            )

        else:

            raise ValueError(
                f"invalid normalization={normalization!r}"
            )

    return scalogram.astype(dtype)
