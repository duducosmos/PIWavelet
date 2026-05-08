from __future__ import annotations

import numpy as np

from piwavelet.transforms import wavelet_coherence
from piwavelet.wavelets.base import BaseWavelet

from .ar1 import estimate_ar1


def generate_ar1(
    alpha: float,
    sigma: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate AR(1) surrogate series.
    """

    noise = rng.normal(
        scale=sigma,
        size=size,
    )

    x = np.zeros(
        size,
        dtype=np.float64,
    )

    for i in range(1, size):
        x[i] = (
            alpha * x[i - 1]
            + noise[i]
        )

    return x


def wcsignificance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dt: float,
    dj: float = 1 / 12,
    s0: float | None = None,
    J: int | None = None,
    wavelet: BaseWavelet,
    significance_level: float = 0.95,
    n_surrogates: int = 300,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Monte Carlo significance test for
    wavelet coherence.

    Follows the approach of:

        Grinsted et al. (2004)

    using AR(1) surrogate pairs.

    Parameters
    ----------
    x, y : np.ndarray
        Input signals.

    dt : float
        Sampling interval.

    dj : float, default=1/12
        Scale spacing.

    s0 : float | None
        Smallest scale.

    J : int | None
        Number of scales.

    wavelet : BaseWavelet
        Mother wavelet.

    significance_level : float, default=0.95
        Confidence level.

    n_surrogates : int, default=300
        Number of Monte Carlo surrogates.

    random_state : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Coherence significance threshold
        for each scale.

        Shape:

            (n_scales,)
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "x and y must be 1D arrays"
        )

    if len(x) != len(y):
        raise ValueError(
            "x and y must have same length"
        )

    if not (0 < significance_level < 1):
        raise ValueError(
            "significance_level must be in (0, 1)"
        )

    if n_surrogates < 10:
        raise ValueError(
            "n_surrogates must be >= 10"
        )

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------
    # estimate AR1 parameters
    # --------------------------------------------------

    ar1_x = estimate_ar1(x)

    ar1_y = estimate_ar1(y)

    # --------------------------------------------------
    # reference transform
    # --------------------------------------------------

    ref = wavelet_coherence(
        x,
        y,
        dt=dt,
        dj=dj,
        s0=s0,
        J=J,
        wavelet=wavelet,
    )

    n_scales = len(ref.scales)

    surrogate_distribution = np.empty(
        (
            n_surrogates,
            n_scales,
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------
    # Monte Carlo loop
    # --------------------------------------------------

    for k in range(n_surrogates):

        xs = generate_ar1(
            alpha=ar1_x.alpha,
            sigma=ar1_x.noise_variance,
            size=len(x),
            rng=rng,
        )

        ys = generate_ar1(
            alpha=ar1_y.alpha,
            sigma=ar1_y.noise_variance,
            size=len(y),
            rng=rng,
        )

        coh = wavelet_coherence(
            xs,
            ys,
            dt=dt,
            dj=dj,
            s0=s0,
            J=J,
            wavelet=wavelet,
        )

        # --------------------------------------------------
        # scale-dependent significance
        #
        # use percentile over time
        # --------------------------------------------------

        surrogate_distribution[k] = np.percentile(
            coh.coherence,
            significance_level * 100,
            axis=1,
        )

    # --------------------------------------------------
    # final significance threshold
    # --------------------------------------------------

    significance = np.percentile(
        surrogate_distribution,
        significance_level * 100,
        axis=0,
    )

    return significance
