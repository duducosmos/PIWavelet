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
class ContinuousWaveletResult:
    """
    Container for CWT results.
    """

    time: np.ndarray

    signal: np.ndarray
    normalized_signal: np.ndarray

    wavelet: np.ndarray

    power: np.ndarray
    global_power: np.ndarray
    fft_power: np.ndarray

    periods: np.ndarray
    frequencies: np.ndarray
    scales: np.ndarray

    coi: np.ndarray

    significance: np.ndarray
    global_significance: np.ndarray

    scale_average: np.ndarray | None = None
    scale_average_significance: float | None = None

    wavelet_name: str = "Morlet"


def plot_wavelet(
    result: ContinuousWaveletResult,
    *,
    title: str,
    signal_label: str,
    units: str = "",
    style: WaveletPlotStyle | None = None,
    show_scale_average: bool = False,
) -> plt.Figure:
    """
    Plot classical Torrence & Compo wavelet figure.
    """

    style = style or WaveletPlotStyle()

    fig = plt.figure(
        figsize=style.figsize,
        dpi=style.dpi,
        constrained_layout=style.constrained_layout,
    )

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

    ax_signal = fig.add_subplot(gs[0, 0])

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

    time = result.time
    periods = result.periods

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    ax_signal.plot(
        time,
        result.normalized_signal,
        color=style.signal_color,
        linewidth=style.linewidth,
    )

    ax_signal.set_title(f"a) {title}")

    if units:
        ax_signal.set_ylabel(f"{signal_label} [{units}]")
    else:
        ax_signal.set_ylabel(signal_label)

    # ------------------------------------------------------------------
    # Wavelet power spectrum
    # ------------------------------------------------------------------

    power = result.power

    if style.use_log2_power:
        power_plot = np.log2(power)
        levels = np.log2(style.power_levels)
    else:
        power_plot = power
        levels = style.power_levels

    cf = ax_power.contourf(
        time,
        np.log2(periods),
        power_plot,
        levels=levels,
        cmap=style.cmap,
        extend="both",
    )

    if style.show_significance:
        ax_power.contour(
            time,
            np.log2(periods),
            result.significance,
            levels=[1.0],
            colors=style.significance_color,
            linewidths=style.significance_linewidth,
        )

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

    ax_power.set_ylabel("Period")

    apply_period_axis_format(
        ax_power,
        periods,
    )

    if style.show_colorbar:
        fig.colorbar(
            cf,
            ax=ax_power,
            pad=0.02,
        )

    # ------------------------------------------------------------------
    # Global spectrum
    # ------------------------------------------------------------------

    ax_global.plot(
        result.global_significance,
        np.log2(periods),
        "--",
        color=style.significance_color,
    )

    ax_global.plot(
        result.fft_power,
        np.log2(1.0 / result.frequencies),
        color=style.fft_color,
    )

    ax_global.plot(
        result.global_power,
        np.log2(periods),
        color=style.spectrum_color,
        linewidth=style.linewidth,
    )

    ax_global.set_title("c) Global Spectrum")

    if units:
        ax_global.set_xlabel(f"Power [{units}²]")
    else:
        ax_global.set_xlabel("Power")

    apply_period_axis_format(
        ax_global,
        periods,
    )

    ax_global.tick_params(
        axis="y",
        labelleft=False,
    )

    # ------------------------------------------------------------------
    # Scale average
    # ------------------------------------------------------------------

    if (
        show_scale_average
        and ax_scale is not None
        and result.scale_average is not None
    ):
        ax_scale.plot(
            time,
            result.scale_average,
            color=style.signal_color,
            linewidth=style.linewidth,
        )

        if result.scale_average_significance is not None:
            ax_scale.axhline(
                result.scale_average_significance,
                linestyle="--",
                color=style.significance_color,
            )

        ax_scale.set_title("d) Scale Averaged Power")
        ax_scale.set_xlabel("Time")

    else:
        ax_power.set_xlabel("Time")

    return fig
