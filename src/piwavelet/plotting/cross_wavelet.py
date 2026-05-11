from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from .styles import WaveletPlotStyle
from .utils import (
    add_coi_overlay,
    apply_period_axis_format,
)


@dataclass(slots=True)
class CrossWaveletResult:
    """
    Result container for cross wavelet transform.
    """

    time: np.ndarray

    cross_wavelet: np.ndarray

    periods: np.ndarray
    frequencies: np.ndarray
    scales: np.ndarray

    coi: np.ndarray

    significance: np.ndarray

    phase: np.ndarray | None = None


def plot_cross_wavelet(
    result: CrossWaveletResult,
    *,
    title: str,
    units: str = "",
    style: WaveletPlotStyle | None = None,
    show_phase: bool = True,
) -> plt.Figure:
    """
    Plot cross-wavelet spectrum.
    """

    style = style or WaveletPlotStyle()

    fig, ax = plt.subplots(
        figsize=style.figsize,
        dpi=style.dpi,
        constrained_layout=style.constrained_layout,
    )

    power = np.abs(result.cross_wavelet)

    # ------------------------------------------------------------------
    # numerical stability
    # ------------------------------------------------------------------

    power = np.maximum(
        power,
        np.finfo(np.float64).eps,
    )

    power_levels = np.asarray(
        style.power_levels,
        dtype=np.float64,
    )

    # remove invalid/non-positive levels
    power_levels = power_levels[
        power_levels > 0
    ]

    if power_levels.size < 2:

        raise ValueError(
            "style.power_levels must contain "
            "at least two positive values"
        )

    # ------------------------------------------------------------------
    # plotting scale
    # ------------------------------------------------------------------

    if style.use_log2_power:

        power_plot = np.log2(
            power
        )

        levels = np.log2(
            power_levels
        )

    else:

        power_plot = power

        levels = power_levels

    # ensure strictly increasing levels
    levels = np.unique(levels)

    if levels.size < 2:

        raise ValueError(
            "contour levels are not valid"
        )

    cf = ax.contourf(
        result.time,
        np.log2(result.periods),
        power_plot,
        levels=levels,
        cmap=style.cmap,
        extend="both",
    )

    if style.show_significance:
        ax.contour(
            result.time,
            np.log2(result.periods),
            result.significance,
            levels=[1.0],
            colors=style.significance_color,
            linewidths=style.significance_linewidth,
        )

    if style.show_coi:
        add_coi_overlay(
            ax=ax,
            time=result.time,
            coi=result.coi,
            max_period=result.periods.max(),
            dt=result.time[1] - result.time[0],
            alpha=style.coi_alpha,
            hatch=style.coi_hatch,
        )

    if show_phase and result.phase is not None:
        angle = result.phase

        u = np.cos(angle)
        v = np.sin(angle)

        sy, sx = style.phase_arrow_stride

        ax.quiver(
            result.time[::sx],
            np.log2(result.periods)[::sy],
            u[::sy, ::sx],
            v[::sy, ::sx],
            units="width",
            angles="uv",
            pivot="mid",
        )

    ax.set_title(title)

    ax.set_xlabel(f"Time ({units})")
    ax.set_ylabel(f"Period ({units})")

    apply_period_axis_format(
        ax,
        result.periods,
    )

    if style.show_colorbar:
        fig.colorbar(cf, ax=ax)

    return fig
