from __future__ import annotations

"""
Historical constants and definitions used in
Torrence & Webster (1999) wavelet smoothing.

These values reproduce the classical MATLAB
wavelet coherence implementation behavior.
"""

# Morlet scale decorrelation factor
#
# Used in scale smoothing:
#
#     width = 0.6 / dj * 2
#
MORLET_SCALE_DECORRELATION: float = 0.6


# Minimum allowed smoothing width
MINIMUM_SCALE_SMOOTH_WIDTH: int = 1
