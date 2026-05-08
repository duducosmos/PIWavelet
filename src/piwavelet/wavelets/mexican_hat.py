from __future__ import annotations

from .dog import DOG


class MexicanHat(DOG):
    """
    Mexican Hat wavelet.

    Notes
    -----
    Equivalent to DOG wavelet with m=2.
    """

    name = "MexicanHat"

    def __init__(self) -> None:
        super().__init__(m=2)

    def time_smoothing_scale(
        self,
        scale: float,
    ) -> float:
        """
        Temporal decorrelation scale for
        wavelet coherence smoothing.
        """

        return scale * 0.6
