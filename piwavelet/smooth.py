#!/usr/bin/env python
# *-* Coding: Utf-8 *-*

from __future__ import division

__authors__ = 'Eduardo dos Santos Pereira'
__data__ = '16/06/2015'
__email__ = 'pereira.somoza@gmail.com'

import os

from oct2py import octave

from numpy import zeros, matrix, ceil, log2, pi, exp
from numpy import isreal, ndarray, ma, arange

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import convolve2d

from .piwavelet import cwt, Morlet


def rect(x, normalize=False):
    if type(x) in [int, float]:
        shape = [x, ]
    elif type(x) in [list, dict]:
        shape = x
    elif type(x) in [ndarray, ma.core.MaskedArray]:
        shape = x.shape
    X = zeros(shape)
    X[0] = X[-1] = 0.5
    X[1:-1] = 1

    if normalize:
        X /= X.sum()

    return X


class smooth:
    """
    This class is an Python interface for the Smoothing matlab
    functions of the package for wavelet,
    cross-wavelet and coherence-wavelet analises profided by
    Aslak Grinsted, John C. Moore and Svetlana Jevrejeva.

    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence

    However, the Continuous wavelet transform of the signal,
    in this class, is a pure python
    function.

    Smoothing as in the appendix of Torrence and Webster
    "Inter decadal changes in the ENSO-Monsoon System" 1998
    used in wavelet coherence calculations.
    Only applicable for the Morlet wavelet.
    """
    def __init__(self, wave, dt, freqs, dj, scale):
        self.wave = wave
        self.dt = dt
        self.period = 1.0 / freqs
        self.freqs = freqs
        self.dj = dj
        self.scale = scale
        HOME = os.path.expanduser('~')
        mFiles = HOME + '/.piwavelet/wtc/'
        self.wtcPath = octave.addpath(mFiles)

    def __call__(self):

        return self.smoothwavelet()

    def smoothwavelet(self):
        """
        Smoothing as in the appendix of Torrence and Webster
        "Inter decadal changes in the ENSO-Monsoon System" 1998
        used in wavelet coherence calculations.
         Only applicable for the Morlet wavelet.
        """

        swave = octave.smoothwavelet(self.wave, self.dt, self.period,
                             self.dj, self.scale)
        return swave

    def smooth(self):
        """Smoothing function used in coherence analysis."""
        # The smoothing is performed by using a filter given by the absolute
        # value of the wavelet function at each scale, normalized to have a
        # total weight of unity, according to suggestions by Torrence &
        # Webster (1999) and bry Grinsted et al. (2004).

        W = self.wave

        m, n = W.shape
        T = zeros([m, n])
        T = matrix(T)
        T = T + 0j
        W = matrix(W)

        # Filter in time.
        npad = int(2 ** ceil(log2(n)))
        k = zeros(npad + 1)
        x = arange(1, int(npad / 2), 1)
        k[1: int(npad / 2)] = x
        k[int(npad / 2) + 2:] = -1 * x[::-1]
        k = 2 * pi * k / npad
        k2 = k ** 2

        snorm = self.scale / self.dt

        for i in range(m):
            F = matrix(exp(-0.5 * (snorm[i] ** 2) * k2))
            smooth = ifft(F.T* fft(W[i, :], n=npad))
            T[i, :] = smooth[0:n]

        if isreal(W).all():
            T = T.real

        # Filter in scale. For the Morlet wavelet it's simply a boxcar with
        # 0.6 width.
        wsize = 0.6 / self.dj * 2
        win = rect(int(round(wsize)), normalize=True)
        T = convolve2d(T, win[:, None], 'same')

        return T

if(__name__ == "__main__"):
    sig = [10, 34, 25, 43, 54, 28, 36, 44, 33, 25, 18, 9, 20]
    wave, scales, freqs, coi, fftOut, fftfreqs = cwt(sig,
                                                      1,
                                                      0.25,
                                                      -1,
                                                      -1,
                                                       Morlet(6.)
                                                      )
    sm = smooth(wave, 1, freqs, 0.25, scales)
    a = sm.smoothwavelet()
    b = sm.smooth()
    print(a[0])
    print(b[0])
