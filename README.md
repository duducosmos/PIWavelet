# PIWavelet

A modern Python library for continuous wavelet analysis, cross-wavelet transforms, and wavelet coherence.

PIWavelet provides a modular and scientifically robust implementation of:

* Continuous Wavelet Transform (CWT)
* Cross-Wavelet Transform (XWT)
* Wavelet Coherence (WTC)
* Torrence & Webster smoothing operators
* Cone of Influence (COI)
* Multiple mother wavelets
* Publication-quality visualization tools

The library is designed for scientific computing, signal analysis, geophysics, astrophysics, climatology, and time-frequency analysis applications.

---

# Features

## Continuous Wavelet Transform

* Complex and real wavelets
* FFT-based implementation
* Scale-frequency conversion
* Cone of Influence estimation
* Multi-scale time-frequency analysis

## Cross-Wavelet Transform

* Cross-wavelet power
* Relative phase analysis
* Shared spectral energy detection

## Wavelet Coherence

* Magnitude-squared wavelet coherence
* Torrence-Webster smoothing operators
* Scale-aware temporal smoothing
* Localized time-frequency coherence analysis
* Phase relationship visualization

## Supported Wavelets

* Morlet
* Paul
* DOG (Derivative of Gaussian)
* Mexican Hat

## Visualization

* Wavelet power spectrum plots
* Cross-wavelet plots
* Wavelet coherence plots
* Cone of Influence overlay
* Phase arrows
* Log-period axis formatting
* Publication-ready figures

---

# Installation

```bash
pip install piwavelet
```

Development installation:

```bash
git clone <repository-url>
cd piwavelet
pip install -e .
```

---

# Quick Start

## Continuous Wavelet Transform

```python
import numpy as np

from piwavelet.transforms import cwt
from piwavelet.wavelets import Morlet

# synthetic signal

dt = 0.25

time = np.arange(0, 512) * dt

signal = (
    np.sin(2 * np.pi * time / 32)
    + 0.5 * np.sin(2 * np.pi * time / 8)
)

result = cwt(
    signal=signal,
    dt=dt,
    wavelet=Morlet(),
)
```

---

## Wavelet Coherence

```python
import numpy as np

from piwavelet.transforms import wavelet_coherence
from piwavelet.wavelets import Morlet

np.random.seed(42)

# ------------------------------------------------------------------
# synthetic signal
# ------------------------------------------------------------------

dt = 0.25

time = np.arange(0, 512) * dt

# localized coherent structure
period = 16

shared = np.sin(
    2 * np.pi * time / period
)

x = shared.copy()

x += 0.4 * np.sin(
    2 * np.pi * time / 6
)

x += 0.5 * np.random.randn(len(time))

window = np.exp(
    -0.5 * ((time - 64) / 5) ** 2
)

y = (
    window * shared
    + 0.8 * np.random.randn(len(time))
)

# ------------------------------------------------------------------
# wavelet coherence
# ------------------------------------------------------------------

result = wavelet_coherence(
    x,
    y,
    dt=dt,
    wavelet=Morlet(),
)
```

This example produces a localized coherence region centered approximately at:

* time ≈ 64
* period ≈ 16

which demonstrates proper time-frequency localization.

---

# Plotting

## Wavelet Coherence Plot

```python
from piwavelet.plotting import plot_wavelet_coherence

fig = plot_wavelet_coherence(
    result,
    title="Wavelet Coherence",
    show_phase=True,
)
```

---

# Scientific Background

PIWavelet follows the classical formulations presented in:

* Torrence, C. & Compo, G. P. (1998)
  "A Practical Guide to Wavelet Analysis"

* Torrence, C. & Webster, P. J. (1999)
  "Interdecadal Changes in the ENSO-Monsoon System"

* Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004)
  "Application of the cross wavelet transform and wavelet coherence to geophysical time series"

Implemented features include:

* scale-normalized coherence
* Torrence-Webster smoothing
* scale-dependent temporal smoothing
* cone of influence estimation
* phase relationship analysis

---

# Architecture

The project is organized into modular components:

```text
piwavelet/
├── transforms/
├── wavelets/
├── smoothing/
├── plotting/
├── significance/
└── utils/
```

---

# Development Goals

Planned and ongoing features:

* Significance testing
* Monte Carlo coherence significance
* Partial wavelet coherence
* Multivariate coherence
* GPU acceleration
* Streaming transforms
* Better statistical diagnostics
* Xarray integration
* Interactive plotting

---

# License

MIT License

---

# Contributing

Contributions, issues, and suggestions are welcome.

Please open an issue or submit a pull request.
