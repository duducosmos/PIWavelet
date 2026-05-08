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
