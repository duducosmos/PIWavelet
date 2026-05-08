from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .models import SignificanceResult
from .protocols import WaveletProtocol
from .spectrum import ar1_spectrum


def _signal_variance(
    signal: np.ndarray | float,
) -> float:
    """
    Return signal variance.

    If a scalar is provided, it is interpreted directly
    as the variance.
    """
    if np.isscalar(signal):
        return float(signal)

    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    return float(np.var(x, ddof=0))


def pointwise_significance(
    signal: np.ndarray | float,
    dt: float,
    scales: np.ndarray,
    alpha: float,
    significance_level: float,
    wavelet: WaveletProtocol,
) -> SignificanceResult:
    """
    Pointwise wavelet significance test.

    Torrence & Compo (1998), equation 18.
    """
    scales = np.asarray(scales, dtype=np.float64)

    variance = _signal_variance(signal)

    period = scales * wavelet.flambda()

    freq = dt / period

    fft_theor = variance * ar1_spectrum(freq, alpha)

    dof = wavelet.dofmin

    chisquare = chi2.ppf(significance_level, dof) / dof

    signif = fft_theor * chisquare

    return SignificanceResult(
        signif=signif,
        fft_theor=fft_theor,
    )


def time_averaged_significance(
    signal: np.ndarray | float,
    dt: float,
    scales: np.ndarray,
    dof: np.ndarray | float,
    alpha: float,
    significance_level: float,
    wavelet: WaveletProtocol,
) -> SignificanceResult:
    """
    Time-averaged wavelet significance test.

    Torrence & Compo (1998), equation 23.
    """
    scales = np.asarray(scales, dtype=np.float64)

    variance = _signal_variance(signal)

    period = scales * wavelet.flambda()

    freq = dt / period

    fft_theor = variance * ar1_spectrum(freq, alpha)

    dof = np.asarray(dof, dtype=np.float64)

    if dof.ndim == 0:
        dof = np.full_like(scales, dof)

    dof = np.maximum(dof, 1.0)

    dof = (
        wavelet.dofmin
        * np.sqrt(
            1.0
            + (dof * dt / wavelet.gamma / scales) ** 2
        )
    )

    dof = np.maximum(dof, wavelet.dofmin)

    chisquare = chi2.ppf(significance_level, dof) / dof

    signif = fft_theor * chisquare

    return SignificanceResult(
        signif=signif,
        fft_theor=fft_theor,
    )


def scale_averaged_significance(
    signal: np.ndarray | float,
    dt: float,
    scales: np.ndarray,
    scale_range: tuple[float, float],
    dj: float,
    alpha: float,
    significance_level: float,
    wavelet: WaveletProtocol,
) -> SignificanceResult:
    """
    Scale-averaged wavelet significance test.

    Torrence & Compo (1998), equations 25-28.
    """
    scales = np.asarray(scales, dtype=np.float64)

    variance = _signal_variance(signal)

    period = scales * wavelet.flambda()

    freq = dt / period

    fft_theor = variance * ar1_spectrum(freq, alpha)

    s1, s2 = scale_range

    sel = np.where(
        (scales >= s1) & (scales <= s2)
    )[0]

    if sel.size == 0:
        raise ValueError(
            f"No valid scales between {s1} and {s2}."
        )

    navg = sel.size

    Savg = 1.0 / np.sum(1.0 / scales[sel])

    Smid = np.exp(
        (np.log(s1) + np.log(s2)) / 2.0
    )

    dof = (
        wavelet.dofmin
        * navg
        * Savg
        / Smid
    ) * np.sqrt(
        1.0
        + (navg * dj / wavelet.deltaj0) ** 2
    )

    fft_avg = Savg * np.sum(
        fft_theor[sel] / scales[sel]
    )

    chisquare = chi2.ppf(significance_level, dof) / dof

    signif = (
        dj
        * dt
        / wavelet.cdelta
        / Savg
    ) * fft_avg * chisquare

    return SignificanceResult(
        signif=np.asarray(signif),
        fft_theor=np.asarray(fft_avg),
    )
