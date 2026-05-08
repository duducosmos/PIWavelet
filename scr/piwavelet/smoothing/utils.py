from __future__ import annotations

import numpy as np


def next_power_of_two(n: int) -> int:
    """
    Return next power of two >= n.

    Parameters
    ----------
    n : int
        Input size.

    Returns
    -------
    int
        Next power of two.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    return 1 << (n - 1).bit_length()


def angular_frequency_vector(
    n: int,
    *,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Angular frequency vector compatible with MATLAB fftfreq behavior.

    Parameters
    ----------
    n : int
        FFT length.

    dt : float, default=1.0
        Sampling interval.

    Returns
    -------
    np.ndarray
        Angular frequencies.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    return 2.0 * np.pi * np.fft.fftfreq(n, d=dt)


def ensure_2d_complex(
    array: np.ndarray,
) -> np.ndarray:
    """
    Ensure array is 2D complex ndarray.

    Parameters
    ----------
    array : np.ndarray

    Returns
    -------
    np.ndarray
    """
    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            "wave array must be 2-dimensional"
        )

    return np.asarray(array, dtype=np.complex128)


def validate_scales(
    scales: np.ndarray,
    expected_size: int,
) -> np.ndarray:
    """
    Validate scale vector.

    Parameters
    ----------
    scales : np.ndarray
        Scale vector.

    expected_size : int
        Expected number of scales.

    Returns
    -------
    np.ndarray
    """
    scales = np.asarray(scales, dtype=np.float64)

    if scales.ndim != 1:
        raise ValueError(
            "scales must be 1-dimensional"
        )

    if scales.size != expected_size:
        raise ValueError(
            "scales size mismatch"
        )

    return scales
