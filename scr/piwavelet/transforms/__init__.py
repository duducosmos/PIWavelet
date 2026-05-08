from .cwt import cwt
from .icwt import icwt
from .xwt import xwt
from .wtc import wavelet_coherence

from .result import (
    CWTResult,
    XWTResult,
    WaveletCoherenceResult,
)

__all__ = [
    "cwt",
    "icwt",
    "xwt",
    "wavelet_coherence",
    "CWTResult",
    "XWTResult",
    "WaveletCoherenceResult",
]
