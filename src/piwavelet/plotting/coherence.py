from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from piwavelet.transforms.result import (
    WaveletCoherenceResult,
)

from .styles import WaveletPlotStyle
from .utils import (
    add_coi_overlay,
    apply_period_axis_format,
)


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

    time = result.time

    periods = result.periods

    y = (
        np.log2(periods)
        if style.use_log2_power
        else periods
    )

    cf = ax.contourf(
        time,
        y,
        coherence,
        levels=style.coherence_levels,
        cmap=style.cmap,
        extend="max",
    )

    if (
        style.show_significance
        and result.significance is not None
    ):

        significance = (
            result.significance[:, np.newaxis]
        )

        sig95 = (
            coherence
            / significance
        )

        ax.contour(
            time,
            y,
            sig95,
            levels=[1.0],
            colors=style.significance_color,
            linewidths=style.significance_linewidth,
        )

    if style.show_coi:
        add_coi_overlay(
            ax=ax,
            time=time,
            coi=result.coi,
            max_period=periods.max(),
            dt=1.0,
            alpha=style.coi_alpha,
            hatch=style.coi_hatch,
        )

    if show_phase and result.phase is not None:

        phase = result.phase

        angle = 0.5 * np.pi - phase

        u = np.cos(angle)
        v = np.sin(angle)

        sy, sx = style.phase_arrow_stride

        ax.quiver(
            time[::sx],
            y[::sy],
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
        periods,
    )

    if style.show_colorbar:
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Wavelet Coherence")

    return fig
