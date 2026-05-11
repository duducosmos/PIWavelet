from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from piwavelet.wavelets.base import BaseWavelet

from .ar1 import estimate_ar1


@dataclass(slots=True)
class WaveletSignificanceResult:
    """
    Wavelet significance test result.
    """

    significance: np.ndarray
    theoretical_spectrum: np.ndarray
    alpha: float


def _ar1_spectrum(
    frequencies: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Theoretical AR(1) power spectrum.

    Follows Torrence & Compo (1998),
    equation (16).
    """

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    return (
        (1.0 - alpha**2)
        / (
            1.0
            + alpha**2
            - 2.0 * alpha * np.cos(
                2.0 * np.pi * frequencies
            )
        )
    )


def wavelet_significance(
    x: np.ndarray | float,
    *,
    dt: float,
    scales: np.ndarray,
    wavelet: BaseWavelet,
    significance_level: float = 0.95,
    sigma_test: int = 0,
    dof: float | np.ndarray | tuple[float, float] | None = None,
    alpha: float | None = None,
) -> WaveletSignificanceResult:
    """
    Significance testing for the continuous
    wavelet transform.

    Implements the methodology of:

        Torrence & Compo (1998)

    Parameters
    ----------
    x : np.ndarray | float
        Input signal.

        If float:
            interpreted as variance.

        If array:
            variance is estimated from data.

    dt : float
        Sampling interval.

    scales : np.ndarray
        Wavelet scales.

    wavelet : BaseWavelet
        Mother wavelet.

    significance_level : float, default=0.95
        Confidence level.

    sigma_test : int, default=0
        Significance test type.

        0:
            regular chi-square test

        1:
            time-averaged test

        2:
            scale-averaged test

    dof : optional
        Degrees of freedom.

    alpha : float | None
        AR(1) coefficient.

        If None:
            estimated from signal.

    Returns
    -------
    WaveletSignificanceResult
    """

    if not (0.0 < significance_level < 1.0):
        raise ValueError(
            "significance_level must be in (0, 1)"
        )

    scales = np.asarray(
        scales,
        dtype=np.float64,
    )

    if scales.ndim != 1:
        raise ValueError(
            "scales must be 1D"
        )

    if np.any(scales <= 0):
        raise ValueError(
            "scales must be positive"
        )

    # --------------------------------------------------
    # variance + length
    # --------------------------------------------------

    if np.isscalar(x):

        variance = float(x)
        n = 1

    else:

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        if x.ndim != 1:
            raise ValueError(
                "x must be 1D"
            )

        variance = np.var(
            x,
            ddof=1,
        )

        n = len(x)

    # --------------------------------------------------
    # AR1 coefficient
    # --------------------------------------------------

    if alpha is None:

        if np.isscalar(x):
            raise ValueError(
                "alpha must be provided "
                "when x is scalar"
            )

        alpha = estimate_ar1(x).alpha

    alpha = float(alpha)

    # --------------------------------------------------
    # scale spacing
    # --------------------------------------------------

    if len(scales) < 2:
        raise ValueError(
            "at least two scales are required"
        )

    dj = np.log2(
        scales[1] / scales[0]
    )

    # --------------------------------------------------
    # wavelet constants
    # --------------------------------------------------

    dofmin = wavelet.dofmin
    cdelta = wavelet.cdelta
    gamma = wavelet.gamma
    deltaj0 = wavelet.deltaj0

    # --------------------------------------------------
    # Fourier periods
    # --------------------------------------------------

    periods = (
        scales
        * wavelet.flambda()
    )

    frequencies = dt / periods

    # --------------------------------------------------
    # theoretical red-noise spectrum
    # --------------------------------------------------

    theoretical_spectrum = (
        variance
        * _ar1_spectrum(
            frequencies,
            alpha,
        )
    )

    significance = theoretical_spectrum.copy()

    # --------------------------------------------------
    # sigma_test = 0
    #
    # no smoothing
    #
    # Torrence & Compo (1998)
    # equation (18)
    # --------------------------------------------------

    if sigma_test == 0:

        dof_eff = dofmin

        chisquare = (
            chi2.ppf(
                significance_level,
                dof_eff,
            )
            / dof_eff
        )

        significance = (
            theoretical_spectrum
            * chisquare
        )

    # --------------------------------------------------
    # sigma_test = 1
    #
    # time-averaged significance
    #
    # Torrence & Compo (1998)
    # equation (23)
    # --------------------------------------------------

    elif sigma_test == 1:

        if dof is None:
            raise ValueError(
                "dof must be provided "
                "for sigma_test=1"
            )

        dof = np.asarray(
            dof,
            dtype=np.float64,
        )

        if dof.ndim == 0:
            dof = np.full(
                len(scales),
                dof,
                dtype=np.float64,
            )

        if len(dof) != len(scales):
            raise ValueError(
                "dof must match number of scales"
            )

        dof = np.maximum(
            dof,
            1.0,
        )

        dof_eff = (
            dofmin
            * np.sqrt(
                1.0
                + (
                    dof
                    * dt
                    / gamma
                    / scales
                )**2
            )
        )

        dof_eff = np.maximum(
            dof_eff,
            dofmin,
        )

        chisquare = (
            chi2.ppf(
                significance_level,
                dof_eff,
            )
            / dof_eff
        )

        significance = (
            theoretical_spectrum
            * chisquare
        )

    # --------------------------------------------------
    # sigma_test = 2
    #
    # scale-averaged significance
    #
    # Torrence & Compo (1998)
    # equations (25-28)
    # --------------------------------------------------

    elif sigma_test == 2:

        if dof is None:
            raise ValueError(
                "dof must be provided "
                "for sigma_test=2"
            )

        if cdelta == -1:
            raise ValueError(
                "cdelta undefined for wavelet"
            )

        if deltaj0 == -1:
            raise ValueError(
                "deltaj0 undefined for wavelet"
            )

        if len(dof) != 2:
            raise ValueError(
                "dof must be (s1, s2)"
            )

        s1, s2 = dof

        selection = (
            (scales >= s1)
            & (scales <= s2)
        )

        if not np.any(selection):
            raise ValueError(
                "no scales in selected interval"
            )

        selected_scales = scales[
            selection
        ]

        navg = len(selected_scales)

        savg = (
            1.0
            / np.sum(
                1.0 / selected_scales
            )
        )

        smid = np.exp(
            (
                np.log(s1)
                + np.log(s2)
            )
            / 2.0
        )

        dof_eff = (
            dofmin
            * navg
            * savg
            / smid
            * np.sqrt(
                1.0
                + (
                    navg
                    * dj
                    / deltaj0
                )**2
            )
        )

        fft_theor = (
            savg
            * np.sum(
                theoretical_spectrum[
                    selection
                ]
                / selected_scales
            )
        )

        chisquare = (
            chi2.ppf(
                significance_level,
                dof_eff,
            )
            / dof_eff
        )

        significance = (
            (
                dj
                * dt
            )
            / (
                cdelta
                * savg
            )
            * fft_theor
            * chisquare
        )

        significance = np.asarray(
            significance,
            dtype=np.float64,
        )

    else:

        raise ValueError(
            "sigma_test must be 0, 1, or 2"
        )

    return WaveletSignificanceResult(
        significance=np.asarray(
            significance,
            dtype=np.float64,
        ),
        theoretical_spectrum=np.asarray(
            theoretical_spectrum,
            dtype=np.float64,
        ),
        alpha=alpha,
    )
