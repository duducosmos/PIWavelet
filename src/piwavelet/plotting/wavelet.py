from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .styles import WaveletPlotStyle
from .utils import (
    add_coi_overlay,
    apply_period_axis_format,
)

from piwavelet.diagnostics import WaveletDiagnostics


def plot_wavelet(
    result: WaveletDiagnostics,
    *,
    title: str,
    signal_label: str,
    units: str = "",
    style: WaveletPlotStyle | None = None,
    show_scale_average: bool = False,
) -> plt.Figure:
    """
    Plot classical Torrence & Compo wavelet analysis figure.

    Layout
    ------
    a) normalized signal
    b) wavelet power spectrum
    c) global wavelet spectrum
    d) scale-averaged spectrum (optional)
    """

    style = style or WaveletPlotStyle()

    fig = plt.figure(
        figsize=style.figsize,
        dpi=style.dpi,
        constrained_layout=style.constrained_layout,
    )

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    if show_scale_average:

        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[4, 1],
            height_ratios=[1, 2, 1],
        )

    else:

        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[4, 1],
            height_ratios=[1, 2],
        )

    ax_signal = fig.add_subplot(
        gs[0, 0]
    )

    ax_power = fig.add_subplot(
        gs[1, 0],
        sharex=ax_signal,
    )

    ax_global = fig.add_subplot(
        gs[1, 1],
        sharey=ax_power,
    )

    ax_scale = None

    if show_scale_average:

        ax_scale = fig.add_subplot(
            gs[2, 0],
            sharex=ax_signal,
        )

    # ------------------------------------------------------------------
    # aliases
    # ------------------------------------------------------------------

    time = result.time

    periods = result.periods

    log2_periods = np.log2(
        periods
    )

    # ------------------------------------------------------------------
    # signal panel
    # ------------------------------------------------------------------

    ax_signal.plot(
        time,
        result.normalized_signal,
        color=style.signal_color,
        linewidth=style.linewidth,
    )

    ax_signal.set_title(
        f"a) {title}"
    )

    if units:

        ax_signal.set_ylabel(
            f"{signal_label} [{units}]"
        )

    else:

        ax_signal.set_ylabel(
            signal_label
        )

    ax_signal.grid(
        visible=style.show_grid,
        alpha=style.grid_alpha,
    )

    # ------------------------------------------------------------------
    # wavelet power spectrum
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

    if power_levels.size < 2:

        raise ValueError(
            "power_levels must contain "
            "at least two positive values"
        )

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

    if levels.size < 2:

        raise ValueError(
            "contour levels must be unique"
        )

    cf = ax_power.contourf(
        time,
        log2_periods,
        power_plot,
        levels=levels,
        cmap=style.cmap,
        extend="both",
    )

    # ------------------------------------------------------------------
    # significance contour
    #
    # Torrence & Compo convention:
    #
    # sig95 > 1 => significant
    # ------------------------------------------------------------------

    if style.show_significance:

        ax_power.contour(
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
            ax=ax_power,
            time=time,
            coi=result.coi,
            max_period=periods.max(),
            dt=time[1] - time[0],
            alpha=style.coi_alpha,
            hatch=style.coi_hatch,
        )

    ax_power.set_title(
        f"b) Wavelet Power Spectrum ({result.wavelet_name})"
    )

    ax_power.set_ylabel(
        "Period"
    )

    apply_period_axis_format(
        ax=ax_power,
        periods=periods,
    )

    ax_power.invert_yaxis()

    ax_power.grid(
        visible=style.show_grid,
        alpha=style.grid_alpha,
    )

    # ------------------------------------------------------------------
    # colorbar
    # ------------------------------------------------------------------

    if style.show_colorbar:

        cbar = fig.colorbar(
            cf,
            ax=ax_power,
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

    # ------------------------------------------------------------------
    # global wavelet spectrum
    # ------------------------------------------------------------------

    global_power = np.maximum(
        result.global_power,
        np.finfo(np.float64).eps,
    )

    ax_global.plot(
        global_power,
        log2_periods,
        color=style.spectrum_color,
        linewidth=style.linewidth,
        label="Wavelet",
    )

    # ------------------------------------------------------------------
    # Fourier spectrum
    # ------------------------------------------------------------------

    positive = (
        result.frequencies > 0
    )

    fft_periods = (
        1.0
        / result.frequencies[
            positive
        ]
    )

    fft_power = np.maximum(
        result.fft_power[
            positive
        ],
        np.finfo(np.float64).eps,
    )

    fft_power = (
        fft_power
        / np.nanmax(fft_power)
        * np.nanmax(global_power)
    )

    ax_global.plot(
        fft_power,
        np.log2(fft_periods),
        color=style.fft_color,
        linewidth=style.linewidth,
        label="Fourier",
    )

    # ------------------------------------------------------------------
    # global significance
    # ------------------------------------------------------------------

    ax_global.plot(
        result.global_significance,
        log2_periods,
        linestyle="--",
        color=style.significance_color,
        linewidth=style.significance_linewidth,
        label="95% Significance",
    )

    ax_global.set_title(
        "c) Global Wavelet Spectrum"
    )

    if units:

        ax_global.set_xlabel(
            f"Power [{units}²]"
        )

    else:

        ax_global.set_xlabel(
            "Power"
        )

    apply_period_axis_format(
        ax=ax_global,
        periods=periods,
    )

    ax_global.tick_params(
        axis="y",
        labelleft=False,
    )

    ax_global.grid(
        visible=style.show_grid,
        alpha=style.grid_alpha,
    )

    # ------------------------------------------------------------------
    # legend
    # ------------------------------------------------------------------

    ax_global.legend(
        loc="best",
        fontsize=style.fontsize * 0.85,
        frameon=False,
    )

    # ------------------------------------------------------------------
    # scale averaged spectrum
    # ------------------------------------------------------------------

    if (
        show_scale_average
        and ax_scale is not None
        and hasattr(
            result,
            "scale_average",
        )
        and result.scale_average is not None
    ):

        ax_scale.plot(
            time,
            result.scale_average,
            color=style.signal_color,
            linewidth=style.linewidth,
        )

        if (
            hasattr(
                result,
                "scale_average_significance",
            )
            and result.scale_average_significance
            is not None
        ):

            ax_scale.axhline(
                result.scale_average_significance,
                linestyle="--",
                color=style.significance_color,
            )

        ax_scale.set_title(
            "d) Scale Averaged Power"
        )

        ax_scale.set_xlabel(
            "Time"
        )

        ax_scale.grid(
            visible=style.show_grid,
            alpha=style.grid_alpha,
        )

    else:

        ax_power.set_xlabel(
            "Time"
        )

    # ------------------------------------------------------------------
    # cosmetics
    # ------------------------------------------------------------------

    plt.setp(
        ax_signal.get_xticklabels(),
        visible=False,
    )

    fig.align_ylabels()

    return fig
