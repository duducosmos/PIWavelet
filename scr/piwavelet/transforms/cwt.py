from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from piwavelet.transforms.common import (
    build_wavelet_ft,
    compute_angular_frequencies,
    compute_coi,
    compute_nfft,
    compute_scales,
    validate_signal,
)

from piwavelet.transforms.frequency import (
    scale_to_frequency,
    scale_to_period,
)

from piwavelet.transforms.result import CWTResult

from piwavelet.wavelets.base import BaseWavelet
from piwavelet.wavelets.morlet import Morlet


def cwt(
    signal: ArrayLike,
    dt: float = 1.0,
    dj: float = 1 / 12,
    s0: float | None = None,
    J: int | None = None,
    wavelet: BaseWavelet | None = None,
) -> CWTResult:
    """
    Continuous Wavelet Transform following
    Torrence & Compo (1998).
    """

    signal = validate_signal(signal)

    if wavelet is None:
        wavelet = Morlet()

    n0 = signal.size

    if s0 is None:
        s0 = 2.0 * dt / wavelet.flambda()

    if J is None:
        J = int(np.log2(n0 * dt / s0) / dj)

    nfft = compute_nfft(n0)

    signal_ft = np.fft.fft(signal, nfft)

    omega = compute_angular_frequencies(nfft, dt)

    scales = compute_scales(s0, dj, J)

    frequencies = scale_to_frequency(scales, wavelet)

    periods = scale_to_period(scales, wavelet)

    W = np.empty(
        (scales.size, nfft),
        dtype=np.complex128,
    )

    for idx, scale in enumerate(scales):

        daughter = build_wavelet_ft(
            wavelet=wavelet,
            scale=scale,
            omega=omega,
            nfft=nfft,
        )

        W[idx] = np.fft.ifft(
            signal_ft * daughter
        )

    coi = compute_coi(
        n_samples=n0,
        dt=dt,
        wavelet=wavelet,
    )

    return CWTResult(
        coefficients=W[:, :n0],
        scales=scales,
        frequencies=frequencies,
        periods=periods,
        coi=coi,
        fft=signal_ft,
        fft_frequencies=omega / (2.0 * np.pi),
        dt=dt,
        dj=dj,
        s0=s0,
        wavelet=wavelet,
    )
