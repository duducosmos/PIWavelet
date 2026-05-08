from __future__ import annotations

import numpy as np


def ar1_spectrum(
    freqs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Theoretical AR(1) power spectrum.

    Parameters
    ----------
    freqs : np.ndarray
        Normalized frequencies.
    alpha : float
        Lag-1 autoregressive coefficient.

    Returns
    -------
    np.ndarray
        Theoretical red-noise spectrum.
    """
    freqs = np.asarray(freqs, dtype=np.float64)

    numerator = 1.0 - alpha**2

    denominator = np.abs(
        1.0 - alpha * np.exp(-2j * np.pi * freqs)
    ) ** 2

    return numerator / denominator
