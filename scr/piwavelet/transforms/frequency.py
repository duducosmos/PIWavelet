from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from piwavelet.wavelets.base import BaseWavelet


def scale_to_frequency(
    scales: NDArray[np.float64],
    wavelet: BaseWavelet,
) -> NDArray[np.float64]:
    return 1.0 / (wavelet.flambda() * scales)


def scale_to_period(
    scales: NDArray[np.float64],
    wavelet: BaseWavelet,
) -> NDArray[np.float64]:
    return wavelet.flambda() * scales


def frequency_to_scale(
    frequencies: NDArray[np.float64],
    wavelet: BaseWavelet,
) -> NDArray[np.float64]:
    return 1.0 / (wavelet.flambda() * frequencies)
