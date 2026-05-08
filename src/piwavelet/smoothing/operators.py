from __future__ import annotations

import numpy as np

from piwavelet.wavelets.base import BaseWavelet

from .scale import smooth_scale
from .time import smooth_time


def smooth_wavelet(
    wave: np.ndarray,
    scales: np.ndarray,
    dt: float,
    dj: float,
    wavelet: BaseWavelet,
) -> np.ndarray:
    """
    Full Torrence-Webster wavelet smoothing operator.

    The smoothing operation is applied in two stages:

        temporal smoothing
            ->
        scale smoothing

    Parameters
    ----------
    wave : np.ndarray
        Wavelet transform with shape:

            (n_scales, n_time)

    scales : np.ndarray
        Wavelet scales.

    dt : float
        Sampling interval.

    dj : float
        Scale spacing.

    wavelet : BaseWavelet
        Mother wavelet.

    Returns
    -------
    np.ndarray
        Smoothed wavelet transform.
    """

    wave = np.asarray(wave)

    if wave.ndim != 2:
        raise ValueError(
            "wave must be a 2D array with shape "
            "(n_scales, n_time)"
        )

    scales = np.asarray(
        scales,
        dtype=np.float64,
    )

    if scales.ndim != 1:
        raise ValueError(
            "scales must be a 1D array"
        )

    if wave.shape[0] != len(scales):
        raise ValueError(
            "wave and scales dimensions "
            "are inconsistent"
        )

    if dt <= 0:
        raise ValueError(
            "dt must be positive"
        )

    if dj <= 0:
        raise ValueError(
            "dj must be positive"
        )

    # --------------------------------------------------
    # temporal smoothing
    # --------------------------------------------------

    smoothed = smooth_time(
        wave=wave,
        scales=scales,
        dt=dt,
        wavelet=wavelet,
    )

    # --------------------------------------------------
    # scale smoothing
    # --------------------------------------------------

    smoothed = smooth_scale(
        wave=smoothed,
        dj=dj,
    )

    return smoothed
