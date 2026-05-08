from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from piwavelet.wavelets.base import BaseWavelet


def validate_signal(signal: ArrayLike) -> NDArray[np.float64]:
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    if signal.size < 2:
        raise ValueError("signal must contain at least two samples")

    return signal


def compute_nfft(n_samples: int) -> int:
    """
    Torrence & Compo compatible FFT padding.
    """
    base2 = int(np.log2(n_samples) + 0.4999)
    return 2 ** (base2 + 1)


def compute_scales(
    s0: float,
    dj: float,
    J: int,
) -> NDArray[np.float64]:
    return s0 * 2.0 ** (np.arange(J + 1) * dj)


def compute_angular_frequencies(
    nfft: int,
    dt: float,
) -> NDArray[np.float64]:
    return 2.0 * np.pi * np.fft.fftfreq(nfft, d=dt)


def compute_coi(
    n_samples: int,
    dt: float,
    wavelet: BaseWavelet,
) -> NDArray[np.float64]:
    coi = (
        n_samples / 2.0
        - np.abs(np.arange(n_samples) - (n_samples - 1) / 2.0)
    )

    return (
        wavelet.flambda()
        * wavelet.coi()
        * dt
        * coi
    )


def build_wavelet_ft(
    wavelet: BaseWavelet,
    scale: float,
    omega: NDArray[np.float64],
    nfft: int,
) -> NDArray[np.complex128]:
    """
    Spectral wavelet daughter function.
    """

    norm = np.sqrt(scale * omega[1] * nfft)

    daughter = norm * np.conj(
        wavelet.psi_ft(scale * omega)
    )

    return daughter.astype(np.complex128)
