from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from piwavelet.statistics import (
    AR1EstimationError,
    estimate_ar1,
)


def generate_ar1(
    alpha: float,
    n: int,
    sigma: float = 1.0,
    seed: int = 1234,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    x = np.zeros(n, dtype=np.float64)

    noise = rng.normal(scale=sigma, size=n)

    for i in range(1, n):
        x[i] = alpha * x[i - 1] + noise[i]

    return x


def test_ar1_returns_valid_result() -> None:
    rng = np.random.default_rng(1234)

    signal = rng.normal(size=512)

    result = estimate_ar1(signal)

    assert np.isfinite(result.alpha)
    assert np.isfinite(result.noise_variance)
    assert np.isfinite(result.mu2)


def test_ar1_white_noise_has_small_alpha() -> None:
    rng = np.random.default_rng(42)

    signal = rng.normal(size=4096)

    result = estimate_ar1(signal)

    assert abs(result.alpha) < 0.1


@pytest.mark.parametrize(
    ("alpha",),
    [
        (0.2,),
        (0.5,),
        (0.8,),
    ],
)
def test_ar1_recovers_known_alpha(alpha: float) -> None:
    signal = generate_ar1(alpha=alpha, n=10000)

    result = estimate_ar1(signal)

    assert_allclose(
        result.alpha,
        alpha,
        atol=0.03,
    )


def test_ar1_rejects_multidimensional_input() -> None:
    x = np.ones((10, 10))

    with pytest.raises(ValueError):
        estimate_ar1(x)


def test_ar1_rejects_short_signal() -> None:
    x = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        estimate_ar1(x)


def test_ar1_constant_signal_raises() -> None:
    x = np.ones(128)

    with pytest.raises(AR1EstimationError):
        estimate_ar1(x)


def test_ar1_regression_reference_values() -> None:
    rng = np.random.default_rng(123)

    signal = rng.normal(size=2048)

    result = estimate_ar1(signal)

    assert_allclose(
        result.alpha,
        -0.0134213372,
        atol=1e-10,
    )

    assert_allclose(
        result.noise_variance,
        1.018732991,
        atol=1e-10,
    )

    assert_allclose(
        result.mu2,
        0.000482991,
        atol=1e-10,
    )
