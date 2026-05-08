from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def compute_log2_period_ticks(periods: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Create standard Torrence & Compo log2 period ticks.
    """

    ymin = np.ceil(np.log2(periods.min()))
    ymax = np.ceil(np.log2(periods.max()))

    ticks = 2 ** np.arange(ymin, ymax + 1)

    return np.log2(ticks), [f"{tick:g}" for tick in ticks]


def sanitize_coi(coi: np.ndarray) -> np.ndarray:
    """
    Avoid invalid log2(0) operations in plotting.
    """

    coi = np.asarray(coi, dtype=float).copy()

    zero_mask = coi <= 0

    if np.any(zero_mask):
        nonzero = coi[coi > 0]

        if nonzero.size == 0:
            coi[:] = 1e-12
        else:
            coi[zero_mask] = nonzero.min() * 0.1

    return coi


def apply_period_axis_format(
    ax: Axes,
    periods: np.ndarray,
) -> None:
    """
    Apply standard wavelet log2 period formatting.
    """

    yticks, labels = compute_log2_period_ticks(periods)

    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)

    ax.set_ylim(np.log2([periods.min(), periods.max()]))
    ax.invert_yaxis()


def add_coi_overlay(
    ax: Axes,
    time: np.ndarray,
    coi: np.ndarray,
    max_period: float,
    dt: float,
    alpha: float = 0.3,
    hatch: str = "x",
) -> None:
    """
    Draw cone of influence overlay.
    """

    coi = sanitize_coi(coi)

    x = np.concatenate(
        [
            time[:1] - dt,
            time,
            time[-1:] + dt,
            time[-1:] + dt,
            time[:1] - dt,
            time[:1] - dt,
        ]
    )

    y = np.log2(
        np.concatenate(
            [
                [1e-12],
                coi,
                [1e-12],
                [max_period],
                [max_period],
                [1e-12],
            ]
        )
    )

    ax.fill(
        x,
        y,
        color="black",
        alpha=alpha,
        hatch=hatch,
    )
