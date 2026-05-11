from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from piwavelet.diagnostics import WaveletDiagnostics

from .styles import WaveletPlotStyle
from .utils import (
    add_coi_overlay,
    apply_period_axis_format,
)


def plot_wavelet_power_spectrum(
    result: WaveletDiagnostics,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    style: WaveletPlotStyle | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot only the wavelet power spectrum.

    Includes:

    - wavelet power
    - significance contour
    - cone of influence
    - colorbar
    """

    style = style or WaveletPlotStyle()

    created_figure = ax is None

    if ax is None:

        fig, ax = plt.subplots(
            figsize=style.figsize,
            dpi=style.dpi,
            constrained_layout=style.constrained_layout,
        )

    else:

        fig = ax.figure

    # ------------------------------------------------------------------
    # aliases
    # ------------------------------------------------------------------

    time = result.time

    periods = result.periods

    log2_periods = np.log2(
        periods
    )

    # ------------------------------------------------------------------
    # power
    # ------------------------------------------------------------------

    power = np.maximum(
        result.power,
        np.finfo(np.float64).eps,
    )

    power_levels = np.asarray(
        style.power_levels,
        dtype=np.float64,
    )

    power_levels = power_levels[
        power_levels > 0
    ]

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

    levels = np.unique(levels)

    cf = ax.contourf(
        time,
        log2_periods,
        power_plot,
        levels=levels,
        cmap=style.cmap,
        extend="both",
    )

    # ------------------------------------------------------------------
    # significance
    # ------------------------------------------------------------------

    if style.show_significance:

        ax.contour(
            time,
            log2_periods,
            result.sig95,
            levels=[1.0],
            colors=style.significance_color,
            linewidths=style.significance_linewidth,
        )

    # ------------------------------------------------------------------
    # cone of influence
    # ------------------------------------------------------------------

    if style.show_coi:

        add_coi_overlay(
            ax=ax,
            time=time,
            coi=result.coi,
            max_period=periods.max(),
            dt=time[1] - time[0],
            alpha=style.coi_alpha,
            hatch=style.coi_hatch,
        )

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------

    if title is not None:

        ax.set_title(title)

    else:

        ax.set_title(
            f"Wavelet Power Spectrum ({result.wavelet_name})"
        )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Period"
    )

    apply_period_axis_format(
        ax=ax,
        periods=periods,
    )

    ax.grid(
        visible=style.show_grid,
        alpha=style.grid_alpha,
    )

    # ------------------------------------------------------------------
    # colorbar
    # ------------------------------------------------------------------

    if style.show_colorbar:

        cbar = fig.colorbar(
            cf,
            ax=ax,
            pad=0.02,
        )

        if style.use_log2_power:

            ticks = np.log2(
                power_levels
            )

            cbar.set_ticks(
                ticks
            )

            cbar.set_ticklabels(
                [
                    str(v)
                    for v in power_levels
                ]
            )

        cbar.set_label(
            "Power"
        )

    return fig, ax
