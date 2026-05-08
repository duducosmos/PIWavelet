from __future__ import annotations

import numpy as np


def rect(
    width: int,
    *,
    normalize: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    MATLAB-compatible rectangular kernel.

    The first and last samples receive half weight:

        [0.5, 1, 1, ..., 1, 0.5]

    This reproduces the original behavior used in
    Torrence & Webster / Grinsted smoothing.

    Parameters
    ----------
    width : int
        Kernel width.

    normalize : bool, default=False
        Normalize kernel to unit sum.

    dtype : np.dtype, default=np.float64
        Output dtype.

    Returns
    -------
    np.ndarray
        Rectangular kernel.
    """
    if width <= 0:
        raise ValueError("width must be positive")

    kernel = np.ones(width, dtype=dtype)

    if width == 1:
        kernel[0] = 1.0
    else:
        kernel[0] = 0.5
        kernel[-1] = 0.5

    if normalize:
        kernel /= kernel.sum()

    return kernel


def boxcar_kernel(
    width: int,
    *,
    normalize: bool = True,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Boxcar smoothing kernel used for scale smoothing.

    Parameters
    ----------
    width : int
        Kernel width.

    normalize : bool, default=True
        Normalize kernel to unit sum.

    dtype : np.dtype, default=np.float64
        Output dtype.

    Returns
    -------
    np.ndarray
        Boxcar kernel.
    """
    return rect(
        width,
        normalize=normalize,
        dtype=dtype,
    )


def gaussian_fft_kernel(
    omega: np.ndarray,
    scale_norm: float,
) -> np.ndarray:
    """
    Gaussian smoothing kernel in Fourier space.

    Implements:

        exp(-0.5 * (scale_norm^2) * omega^2)

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency vector.

    scale_norm : float
        Normalized scale:

            scale / dt

    Returns
    -------
    np.ndarray
        Fourier-domain Gaussian kernel.
    """
    omega = np.asarray(omega, dtype=np.float64)

    return np.exp(
        -0.5 * (scale_norm**2) * (omega**2)
    )
