class AR1EstimationError(RuntimeError):
    """
    Raised when AR(1) estimation becomes numerically unstable.
    """


class SignificanceError(RuntimeError):
    """
    Raised when wavelet significance computation fails.
    """
