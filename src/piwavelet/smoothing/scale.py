from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

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
    Scale-direction Gaussian smoothing.
    """

    wave = ensure_2d_complex(wave)

    if dj <= 0:
        raise ValueError(
            "dj must be positive"
        )

    sigma = (
        MORLET_SCALE_DECORRELATION
        / dj
    )

    width = int(
        np.ceil(6 * sigma)
    )

    width = max(
        width,
        MINIMUM_SCALE_SMOOTH_WIDTH,
    )

    if width % 2 == 0:
        width += 1

    x = np.arange(width) - width // 2

    kernel = np.exp(
        -0.5 * (x / sigma) ** 2
    )

    kernel /= kernel.sum()

    smoothed = convolve2d(
        wave,
        kernel[:, None],
        mode="same",
        boundary="symm",
    )

    return smoothed
