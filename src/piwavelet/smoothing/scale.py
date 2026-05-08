from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from .kernels import boxcar_kernel
from .torrence_webster import (
    MINIMUM_SCALE_SMOOTH_WIDTH,
    MORLET_SCALE_DECORRELATION,
)
from .utils import ensure_2d_complex


def smooth_scale(
    wave: np.ndarray,
    dj: float,
) -> np.ndarray:
    """
    Scale-direction smoothing.

    Parameters
    ----------
    wave : np.ndarray
        Wavelet array with shape:

            (n_scales, n_time)

    dj : float
        Scale spacing.

    Returns
    -------
    np.ndarray
        Scale-smoothed wavelet transform.
    """
    wave = ensure_2d_complex(wave)

    if dj <= 0:
        raise ValueError("dj must be positive")

    width = int(
        round(
            (MORLET_SCALE_DECORRELATION / dj) * 2.0
        )
    )

    width = max(
        width,
        MINIMUM_SCALE_SMOOTH_WIDTH,
    )

    kernel = boxcar_kernel(
        width,
        normalize=True,
    )

    smoothed = convolve2d(
        wave,
        kernel[:, None],
        mode="same",
    )

    if np.isrealobj(wave):
        smoothed = smoothed.real

    return smoothed
