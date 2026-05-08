from __future__ import annotations

import numpy as np

from .exceptions import AR1EstimationError
from .models import AR1Result


def estimate_ar1(signal: np.ndarray) -> AR1Result:
    """
    Estimate lag-1 autoregressive parameters using the
    Allen & Smith (1996) unbiased estimator.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional time series.

    Returns
    -------
    AR1Result
        Estimated AR(1) parameters.

    Notes
    -----
    Implements the estimator used by Torrence & Compo (1998)
    and Grinsted et al. (2004).
    """
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    n = x.size

    if n < 3:
        raise ValueError("signal length must be >= 3")

    x = x - np.mean(x)

    # Lag-0 covariance
    c0 = np.dot(x, x) / n

    # Lag-1 covariance
    c1 = np.dot(x[:-1], x[1:]) / (n - 1)

    # Allen & Smith substitutions
    A = c0 * n**2

    B = (
        -c1 * n
        - c0 * n**2
        - 2.0 * c0
        + 2.0 * c1
        - c1 * n**2
        + c0 * n
    )

    C = n * (c0 + c1 * n - c1)

    discriminant = B**2 - 4.0 * A * C

    if discriminant <= 0:
        raise AR1EstimationError(
            "Cannot determine unbiased AR(1) coefficient. "
            "Series may be too short or strongly trended."
        )

    alpha = (-B - np.sqrt(discriminant)) / (2.0 * A)

    # Allen & Smith (1996), footnote 4
    mu2 = (
        -1.0 / n
        + (2.0 / n**2)
        * (
            (n - alpha**n) / (1.0 - alpha)
            - alpha * (1.0 - alpha ** (n - 1)) / (1.0 - alpha) ** 2
        )
    )

    corrected_variance = c0 / (1.0 - mu2)

    noise_variance = np.sqrt((1.0 - alpha**2) * corrected_variance)

    return AR1Result(
        alpha=float(alpha),
        noise_variance=float(noise_variance),
        mu2=float(mu2),
    )
