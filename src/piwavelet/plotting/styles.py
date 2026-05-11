from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class WaveletPlotStyle:
    """
    Visual configuration for wavelet plots.

    This object contains ONLY rendering-related parameters.
    No mathematical parameters belong here.
    """

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------

    figsize: tuple[float, float] = (11.0, 8.0)

    legend_fontsize: float = 10

    fontsize: float = 10

    dpi: int = 120

    constrained_layout: bool = True

    # ------------------------------------------------------------------
    # colormaps
    # ------------------------------------------------------------------

    cmap: str = "viridis"

    # ------------------------------------------------------------------
    # contour levels
    # ------------------------------------------------------------------

    power_levels: Sequence[float] = field(
        default_factory=lambda: (
            0.0,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
        )
    )

    coherence_levels: Sequence[float] = field(
        default_factory=lambda: (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        )
    )

    # ------------------------------------------------------------------
    # line styling
    # ------------------------------------------------------------------

    linewidth: float = 1.5

    significance_linewidth: float = 1.5

    # ------------------------------------------------------------------
    # colors
    # ------------------------------------------------------------------

    signal_color: str = "black"

    spectrum_color: str = "black"

    fft_color: str = "0.6"

    significance_color: str = "black"

    # ------------------------------------------------------------------
    # cone of influence
    # ------------------------------------------------------------------

    coi_alpha: float = 0.3

    coi_hatch: str = "x"

    # ------------------------------------------------------------------
    # typography
    # ------------------------------------------------------------------

    title_fontsize: int = 14

    label_fontsize: int = 12

    tick_fontsize: int = 10

    # ------------------------------------------------------------------
    # scaling
    # ------------------------------------------------------------------

    use_log2_power: bool = True

    use_log2_period: bool = True

    # ------------------------------------------------------------------
    # visibility
    # ------------------------------------------------------------------

    show_coi: bool = True

    show_significance: bool = True

    show_colorbar: bool = True

    show_grid: bool = True

    # ------------------------------------------------------------------
    # grid styling
    # ------------------------------------------------------------------

    grid_alpha: float = 0.25

    # ------------------------------------------------------------------
    # phase arrows
    # ------------------------------------------------------------------

    phase_arrow_stride: tuple[int, int] = (3, 3)
