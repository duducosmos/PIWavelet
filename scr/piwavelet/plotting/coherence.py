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
class WaveletCoherenceResult:
    """
    Result container for wavelet coherence.
    """

    time: np.ndarray

    coherence: np.ndarray

    periods: np.ndarray
    frequencies: np.ndarray
    scales: np.ndarray

    coi: np.ndarray

    significance: np.ndarray

    phase: np.ndarray | None = None


def plot_wavelet_coherence(
    result: WaveletCoherenceResult,
    *,
    title: str,
    units: str = "",
    style: WaveletPlotStyle | None = None,
    show_phase: bool = False,
) -> plt.Figure:
    """
    Plot wavelet coherence.
    """

    style = style or WaveletPlotStyle()

    fig, ax = plt.subplots(
        figsize=style.figsize,
        dpi=style.dpi,
        constrained_layout=style.constrained_layout,
    )

    coherence = result.coherence

    if style.use_log2_power:
        coherence_plot = coherence
        levels = style.coherence_levels
    else:
        coherence_plot = coherence
        levels = style.coherence_levels

    cf = ax.contourf(
        result.time,
        np.log2(result.periods),
        coherence_plot,
        levels=levels,
        cmap=style.cmap,
        extend="max",
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
        angle = 0.5 * np.pi - result.phase

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
