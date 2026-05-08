from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from piwavelet.transforms.cwt import CWTResult


@dataclass(slots=True)
class WaveletDiagnostics:
    """
    Derived diagnostics for classical wavelet analysis plots.

    This object contains quantities derived from a CWTResult
    that are commonly used in Torrence & Compo style figures.

    It intentionally separates:

        - raw transform output
        - derived spectral diagnostics
        - statistical significance

    so that plotting does not depend directly on the
    transform implementation.
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
    # placeholders until proper statistical module exists
    # ------------------------------------------------------------------

    significance: NDArray[np.float64]

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
    Compute derived diagnostics from a Continuous Wavelet Transform.

    Parameters
    ----------
    result
        Continuous Wavelet Transform result.

    signal
        Original signal.

        If omitted, uses result.signal.

    normalized_signal
        Normalized signal used for plotting.

        If omitted, signal is z-score normalized.

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

            normalized_signal = np.zeros_like(signal)

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

        time = result.time

    else:

        time = (
            np.arange(result.n_original)
            * result.dt
        )

    # ------------------------------------------------------------------
    # wavelet power spectrum
    # ------------------------------------------------------------------

    power = np.abs(
        result.coefficients
    ) ** 2

    # ------------------------------------------------------------------
    # global wavelet spectrum
    # Torrence & Compo:
    #
    # global_ws(scale) = mean_t(power)
    # ------------------------------------------------------------------

    global_power = power.mean(axis=1)

    # ------------------------------------------------------------------
    # FFT spectrum
    #
    # interpolate FFT power onto wavelet frequencies
    # for comparison in global spectrum panel
    # ------------------------------------------------------------------

    positive = result.fft_frequencies > 0

    fft_frequencies = (
        result.fft_frequencies[positive]
    )

    fft_power_positive = (
        np.abs(result.fft[positive]) ** 2
    )

    fft_power = np.interp(
        result.frequencies,
        fft_frequencies,
        fft_power_positive,
        left=np.nan,
        right=np.nan,
    )

    # ------------------------------------------------------------------
    # placeholder significance
    #
    # replaced later by:
    #   statistics.significance.wavelet_significance()
    # ------------------------------------------------------------------

    significance = np.ones_like(
        power,
        dtype=np.float64,
    )

    global_significance = np.ones_like(
        result.periods,
        dtype=np.float64,
    )

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
        global_significance=global_significance,
        wavelet_name=type(result.wavelet).__name__,
    )
