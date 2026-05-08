from __future__ import annotations

import numpy as np

from piwavelet.wavelets.base import BaseWavelet

from .kernels import gaussian_fft_kernel
from .utils import (
    angular_frequency_vector,
    ensure_2d_complex,
    validate_scales,
)


def smooth_time(
    wave: np.ndarray,
    scales: np.ndarray,
    dt: float,
    wavelet: BaseWavelet,
) -> np.ndarray:
    """
    Temporal wavelet smoothing in Fourier space.

    Parameters
    ----------
    wave : np.ndarray
        Wavelet transform with shape:

            (n_scales, n_time)

    scales : np.ndarray
        Wavelet scales.

    dt : float
        Sampling interval.

    wavelet : BaseWavelet
        Mother wavelet.

    Returns
    -------
    np.ndarray
        Temporally smoothed transform.
    """

    wave = ensure_2d_complex(wave)

    n_scales, n_time = wave.shape

    scales = validate_scales(
        scales,
        n_scales,
    )

    if dt <= 0:
        raise ValueError(
            "dt must be positive"
        )

    omega = angular_frequency_vector(
        n_time,
        dt=dt,
    )

    output = np.empty_like(
        wave,
        dtype=np.complex128,
    )

    # reflection padding to avoid circular leakage
    pad = n_time // 2

    for idx, scale in enumerate(scales):

        padded = np.pad(
            wave[idx],
            pad_width=pad,
            mode="reflect",
        )

        n_padded = len(padded)

        omega_pad = angular_frequency_vector(
            n_padded,
            dt=dt,
        )

        # temporal decorrelation scale
        decorrelation = (
            wavelet.time_smoothing_scale(scale)
        )

        kernel = gaussian_fft_kernel(
            omega_pad,
            decorrelation,
        )

        spectrum = np.fft.fft(
            padded
        )

        smoothed = np.fft.ifft(
            kernel * spectrum
        )

        output[idx] = smoothed[
            pad:pad + n_time
        ]

    return output
