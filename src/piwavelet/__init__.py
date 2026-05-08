"""
PiWavelet 2

Pure Python wavelet, cross-wavelet and wavelet coherence analysis.
"""

from .transforms.cwt import cwt
from .transforms.icwt import icwt

from .wavelets.morlet import Morlet
from .wavelets.paul import Paul
from .wavelets.dog import DOG
from .wavelets.mexican_hat import MexicanHat

__all__ = [
    "cwt",
    "icwt",
    "Morlet",
    "Paul",
    "DOG",
    "MexicanHat",
]

__version__ = "2.0.0a1"
