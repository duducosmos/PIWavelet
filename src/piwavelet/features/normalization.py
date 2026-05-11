from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def minmax_normalize(
    x: NDArray,
    *,
    eps: float = 1e-12,
) -> NDArray:
    """
    Min-max normalization to [0, 1].
    """

    x = np.asarray(x)

    xmin = np.nanmin(x)

    xmax = np.nanmax(x)

    scale = xmax - xmin

    if scale < eps:

        return np.zeros_like(
            x,
            dtype=np.float64,
        )

    return (
        x - xmin
    ) / scale


def standardize(
    x: NDArray,
    *,
    eps: float = 1e-12,
) -> NDArray:
    """
    Standard score normalization.
    """

    x = np.asarray(x)

    mean = np.nanmean(x)

    std = np.nanstd(x)

    if std < eps:

        return np.zeros_like(
            x,
            dtype=np.float64,
        )

    return (
        x - mean
    ) / std


def robust_normalize(
    x: NDArray,
    *,
    eps: float = 1e-12,
) -> NDArray:
    """
    Median/IQR normalization.
    """

    x = np.asarray(x)

    median = np.nanmedian(x)

    q25 = np.nanpercentile(
        x,
        25,
    )

    q75 = np.nanpercentile(
        x,
        75,
    )

    iqr = q75 - q25

    if iqr < eps:

        return np.zeros_like(
            x,
            dtype=np.float64,
        )

    return (
        x - median
    ) / iqr
