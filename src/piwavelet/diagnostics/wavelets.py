from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from piwavelet.transforms.result import CWTResult


@dataclass(slots=True)
class WaveletDiagnostics:
    """
    Derived diagnostics for classical wavelet analysis.

    This container stores quantities commonly used in
    Torrence & Compo style visualizations while keeping
    plotting decoupled from transform internals.
    """

    # ------------------------------------------------------------------
    # original domain
    # ------------------------------------------------------------------

    time: NDArray[np.float64]

    signal: NDArray[np.float64]

    normalized_signal: NDArray[np.float64]

    # ------------------------------------------------------------------
    # transform domain
    # ------------------------------------------------------------------

    coefficients: NDArray[np.complex128]

    power: NDArray[np.float64]

    # ------------------------------------------------------------------
    # spectral summaries
    # ------------------------------------------------------------------

    global_power: NDArray[np.float64]

    fft_power: NDArray[np.float64]

    # ------------------------------------------------------------------
    # spectral coordinates
    # ------------------------------------------------------------------

    periods: NDArray[np.float64]

    frequencies: NDArray[np.float64]

    scales: NDArray[np.float64]

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    coi: NDArray[np.float64]

    # ------------------------------------------------------------------
    # significance
    # ------------------------------------------------------------------

    significance: NDArray[np.float64]

    sig95: NDArray[np.float64]

    significant: NDArray[np.bool_]

    global_significance: NDArray[np.float64]

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    wavelet_name: str


def compute_wavelet_diagnostics(
    result: CWTResult,
    *,
    signal: NDArray[np.float64] | None = None,
    normalized_signal: NDArray[np.float64] | None = None,
) -> WaveletDiagnostics:
    """
    Compute derived diagnostics from a CWTResult.

    Parameters
    ----------
    result
        Continuous Wavelet Transform result.

    signal
        Original signal.

        If omitted, uses result.signal.

    normalized_signal
        Signal normalized for visualization.

        If omitted, z-score normalization is applied.

    Returns
    -------
    WaveletDiagnostics
    """

    # ------------------------------------------------------------------
    # signal handling
    # ------------------------------------------------------------------

    if signal is None:
        signal = result.signal

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if normalized_signal is None:

        std = signal.std()

        if std == 0.0:

            normalized_signal = np.zeros_like(
                signal
            )

        else:

            normalized_signal = (
                signal - signal.mean()
            ) / std

    normalized_signal = np.asarray(
        normalized_signal,
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # time axis
    # ------------------------------------------------------------------

    if result.time is not None:

        time = np.asarray(
            result.time,
            dtype=np.float64,
        )

    else:

        time = (
            np.arange(
                result.n_original,
                dtype=np.float64,
            )
            * result.dt
        )

    # ------------------------------------------------------------------
    # wavelet power spectrum
    # ------------------------------------------------------------------

    power = result.power

    # ------------------------------------------------------------------
    # global wavelet spectrum
    #
    # Torrence & Compo:
    #
    # global_ws(scale) = mean_t(power)
    # ------------------------------------------------------------------

    global_power = power.mean(
        axis=1
    )

    # ------------------------------------------------------------------
    # FFT frequencies
    #
    # Reconstructed from stored FFT size
    # to avoid padding inconsistencies.
    # ------------------------------------------------------------------

    fft_frequencies = np.fft.fftfreq(
        result.fft.size,
        d=result.dt,
    )

    positive = fft_frequencies > 0

    fft_frequencies = fft_frequencies[
        positive
    ]

    # ------------------------------------------------------------------
    # FFT power spectrum
    #
    # normalized by original signal length
    # ------------------------------------------------------------------

    fft_power_positive = (
        np.abs(
            result.fft[positive]
        ) ** 2
    ) / result.n_original

    # ------------------------------------------------------------------
    # interpolate FFT spectrum onto
    # wavelet frequencies
    # ------------------------------------------------------------------

    fft_power = np.interp(
        result.frequencies,
        fft_frequencies,
        fft_power_positive,
        left=np.nan,
        right=np.nan,
    )

    # ------------------------------------------------------------------
    # significance
    # ------------------------------------------------------------------

    significance = result.significance

    sig95 = result.sig95

    significant = result.significant

    # ------------------------------------------------------------------
    # global significance
    #
    # placeholder approximation:
    # same scale-dependent threshold
    #
    # later:
    # use sigma_test=1
    # ------------------------------------------------------------------

    global_significance = significance

    # ------------------------------------------------------------------
    # diagnostics container
    # ------------------------------------------------------------------

    return WaveletDiagnostics(
        time=time,
        signal=signal,
        normalized_signal=normalized_signal,
        coefficients=result.coefficients,
        power=power,
        global_power=global_power,
        fft_power=fft_power,
        periods=result.periods,
        frequencies=result.frequencies,
        scales=result.scales,
        coi=result.coi,
        significance=significance,
        sig95=sig95,
        significant=significant,
        global_significance=global_significance,
        wavelet_name=type(
            result.wavelet
        ).__name__,
    )
