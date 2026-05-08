from __future__ import annotations

import numpy as np

from .kernels import gaussian_fft_kernel
from .utils import (
    angular_frequency_vector,
    ensure_2d_complex,
    next_power_of_two,
    validate_scales,
)


def smooth_time(
    wave: np.ndarray,
    scales: np.ndarray,
    dt: float,
    *,
    pad_to_power_of_two: bool = True,
) -> np.ndarray:
    """
    Temporal Gaussian smoothing in Fourier space.

    Parameters
    ----------
    wave : np.ndarray
        Complex wavelet transform with shape:

            (n_scales, n_time)

    scales : np.ndarray
        Wavelet scales.

    dt : float
        Sampling interval.

    pad_to_power_of_two : bool, default=True
        Pad time dimension to next power of two.

    Returns
    -------
    np.ndarray
        Temporally smoothed wavelet transform.
    """
    wave = ensure_2d_complex(wave)

    n_scales, n_time = wave.shape

    scales = validate_scales(scales, n_scales)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if pad_to_power_of_two:
        n_fft = next_power_of_two(n_time)
    else:
        n_fft = n_time

    omega = angular_frequency_vector(
        n_fft,
        dt=dt,
    )

    output = np.empty(
        (n_scales, n_time),
        dtype=np.complex128,
    )

    for idx, scale in enumerate(scales):

        scale_norm = scale / dt

        kernel = gaussian_fft_kernel(
            omega,
            scale_norm,
        )

        spectrum = np.fft.fft(
            wave[idx],
            n=n_fft,
        )

        smoothed = np.fft.ifft(
            kernel * spectrum,
            n=n_fft,
        )

        output[idx] = smoothed[:n_time]

    if np.isrealobj(wave):
        output = output.real

    return output
