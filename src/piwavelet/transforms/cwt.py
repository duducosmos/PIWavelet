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

from piwavelet.statistics.wavelet_significance import (
    wavelet_significance,
)


def cwt(
    signal: ArrayLike,
    dt: float = 1.0,
    dj: float = 1 / 12,
    s0: float | None = None,
    J: int | None = None,
    wavelet: BaseWavelet | None = None,
    time: ArrayLike | None = None,
) -> CWTResult:
    """
    Continuous Wavelet Transform following
    Torrence & Compo (1998).
    """

    signal = validate_signal(signal)

    if wavelet is None:
        wavelet = Morlet()

    n0 = signal.size

    # ------------------------------------------------------------------
    # time vector
    # ------------------------------------------------------------------

    if time is not None:

        time = np.asarray(
            time,
            dtype=np.float64,
        )

        if time.ndim != 1:
            raise ValueError(
                "time must be one-dimensional"
            )

        if time.size != n0:
            raise ValueError(
                "time and signal must "
                "have the same length"
            )

    # ------------------------------------------------------------------
    # smallest scale
    # ------------------------------------------------------------------

    if s0 is None:

        s0 = (
            2.0
            * dt
            / wavelet.flambda()
        )

    # ------------------------------------------------------------------
    # number of scales
    # ------------------------------------------------------------------

    if J is None:

        J = int(
            np.log2(
                n0 * dt / s0
            )
            / dj
        )

    # ------------------------------------------------------------------
    # FFT padding
    # ------------------------------------------------------------------

    nfft = compute_nfft(n0)

    # ------------------------------------------------------------------
    # FFT of signal
    # ------------------------------------------------------------------

    signal_ft = np.fft.fft(
        signal,
        nfft,
    )

    # ------------------------------------------------------------------
    # angular frequencies
    # ------------------------------------------------------------------

    omega = compute_angular_frequencies(
        nfft=nfft,
        dt=dt,
    )

    # ------------------------------------------------------------------
    # scales
    # ------------------------------------------------------------------

    scales = compute_scales(
        s0=s0,
        dj=dj,
        J=J,
    )

    frequencies = scale_to_frequency(
        scales=scales,
        wavelet=wavelet,
    )

    periods = scale_to_period(
        scales=scales,
        wavelet=wavelet,
    )

    # ------------------------------------------------------------------
    # wavelet transform
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # truncate padded coefficients
    # ------------------------------------------------------------------

    coefficients = W[:, :n0]

    # ------------------------------------------------------------------
    # wavelet power spectrum
    # ------------------------------------------------------------------

    power = np.abs(
        coefficients
    ) ** 2

    # ------------------------------------------------------------------
    # significance
    # ------------------------------------------------------------------

    significance_result = wavelet_significance(
        x=signal,
        dt=dt,
        scales=scales,
        wavelet=wavelet,
        significance_level=0.95,
    )

    significance = (
        significance_result.significance
    )

    # ------------------------------------------------------------------
    # normalized significance ratio
    #
    # Torrence & Compo convention:
    #
    # sig95 > 1 => significant
    # ------------------------------------------------------------------

    sig95 = (
        power
        / significance[:, np.newaxis]
    )

    significant = sig95 > 1.0

    # ------------------------------------------------------------------
    # cone of influence
    # ------------------------------------------------------------------

    coi = compute_coi(
        n_samples=n0,
        dt=dt,
        wavelet=wavelet,
    )

    # ------------------------------------------------------------------
    # result
    # ------------------------------------------------------------------

    return CWTResult(
        signal=signal,
        time=time,
        coefficients=coefficients,
        power=power,
        significance=significance,
        sig95=sig95,
        significant=significant,
        scales=scales,
        frequencies=frequencies,
        periods=periods,
        fft=signal_ft[:n0],
        angular_frequencies=omega,
        coi=coi,
        dt=dt,
        dj=dj,
        s0=s0,
        n_original=n0,
        n_padded=nfft,
        wavelet=wavelet,
    )
