"""
Plotting API for PiWavelet.

This package contains ONLY visualization utilities.

No mathematical computation should happen here.
Plotting functions consume already-computed transform results.
"""

from .styles import WaveletPlotStyle

from .wavelet import (
    ContinuousWaveletResult,
    plot_wavelet,
)

from .coherence import (
    WaveletCoherenceResult,
    plot_wavelet_coherence,
)

from .cross_wavelet import (
    CrossWaveletResult,
    plot_cross_wavelet,
)

__all__ = [
    # styles
    "WaveletPlotStyle",
    # wavelet
    "ContinuousWaveletResult",
    "plot_wavelet",
    # coherence
    "WaveletCoherenceResult",
    "plot_wavelet_coherence",
    # cross wavelet
    "CrossWaveletResult",
    "plot_cross_wavelet",
]
