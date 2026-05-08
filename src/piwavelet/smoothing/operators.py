from __future__ import annotations

import numpy as np

from .scale import smooth_scale
from .time import smooth_time


def smooth_wavelet(
    wave: np.ndarray,
    scales: np.ndarray,
    dt: float,
    dj: float,
    *,
    pad_to_power_of_two: bool = True,
) -> np.ndarray:
    """
    Full Torrence-Webster wavelet smoothing operator.

    The operation order is:

        temporal smoothing
            ->
        scale smoothing

    Parameters
    ----------
    wave : np.ndarray
        Wavelet transform.

    scales : np.ndarray
        Wavelet scales.

    dt : float
        Sampling interval.

    dj : float
        Scale spacing.

    pad_to_power_of_two : bool, default=True
        Use FFT padding.

    Returns
    -------
    np.ndarray
        Smoothed wavelet transform.
    """
    smoothed = smooth_time(
        wave,
        scales,
        dt,
        pad_to_power_of_two=pad_to_power_of_two,
    )

    smoothed = smooth_scale(
        smoothed,
        dj,
    )

    return smoothed
