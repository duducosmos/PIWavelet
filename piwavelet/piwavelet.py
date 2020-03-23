#!/usr/bin/env python
# -*- Coding: UTF-8 -*-
__name = 'piwavelets'
__authors = 'Eduardo dos Santos Pereira, Regla D. Somoza'
__data = '13/03/2013'
__email = 'pereira.somoza@gmail.com,duthit@gmail.com'

"""
Python Interface for Wavelet  Analises
This module references to the numpy, scipy, pylab and oct2py Python
packages.

DISCLAIMER
    This module is an Python interface for the some matlab functions of the package for wavelet,
    cross-wavelet and coherence-wavelet analises profided by
    Aslak Grinsted, John C. Moore and Svetlana Jevrejeva.

    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence

    However, the Continuous wavelet transform of the signal, in this class, is a pure python
    function.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

TO RUN
    It is necessary to have gnuoctave and oct2py installed in your machine
    Gnu Octave: http://www.gnu.org/software/octave/ (sudo apt-get install octave)
    oct2py: https://github.com/blink1073/oct2py (easy_install oct2py)

AUTHOR
    Eduardo S. Pereira
    email: pereira.somoza@gmail.com


REFERENCES
    [1] Mallat, Stephane G. (1999). A wavelet tour of signal processing
    [2] Addison, Paul S. The illustrated wavelet transform handbook
    [3] Torrence, Christopher and Compo, Gilbert P. (1998). A Practical
            Guide to Wavelet Analysis
    [4] Grinsted, A., Moore, J.C., Jevrejeva, S., 2004,
        Nonlin. Processes Geophys., 11, 561
    [5] Jevrejeva, S., Moore, J.C.,  Grinsted, A., 2003,
         J. Geophys. Res., 108(D21), 4677,
    [6] Torrence, C., Webster, P. ,1999,
            J.Clim., 12, 2679

"""

import os, sys
import numpy

from numpy import (arange, ceil, concatenate, conjugate, cos, exp, isnan, log,
                   log2, ones, pi, prod, real, sqrt, zeros, polyval, array, fix, dtype, modf,
                   around, meshgrid, isreal, round, intersect1d, asarray, matrix)
from numpy.fft import fft, ifft, fftfreq
from numpy.lib.polynomial import polyval
from numpy import abs as nAbs
from numpy import argwhere
from scipy.special import gamma
from scipy.stats import chi2
from scipy.special.orthogonal import hermitenorm
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from oct2py import octave

import pylab
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import  colors
import matplotlib.dates as mdates


############################################################################
############################################################################
############################################################################

"""
Continuous wavelet transform module for Python. Includes a collection
of routines for wavelet transform and statistical analysis via FFT
algorithm. This module references to the numpy, scipy and pylab Python
packages.
Adapted from :
            https://github.com/regeirk/kPyWavelet
            Sebastian Krieger
            email: sebastian@nublia.com

DISCLAIMER
    This module is based on routines provided by C. Torrence and G.
    Compo available at http://paos.colorado.edu/research/wavelets/
    and on routines provided by A. Brazhe available at
    http://cell.biophys.msu.ru/static/swan/.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.



REFERENCES
    [1] Mallat, Stephane G. (1999). A wavelet tour of signal processing
    [2] Addison, Paul S. The illustrated wavelet transform handbook
    [3] Torrence, Christopher and Compo, Gilbert P. (1998). A Practical
        Guide to Wavelet Analysis
"""


#from pylab import find

find = lambda x: argwhere(x)

class Morlet:
    """Implements the Morlet wavelet class.

    Note that the input parameters f and f0 are angular frequencies.
    f0 should be more than 0.8 for this function to be correct, its
    default value is f0=6.

    """

    name = 'Morlet'

    def __init__(self, f0=6.0):
        self._set_f0(f0)

    def psi_ft(self, f):
        """Fourier transform of the approximate Morlet wavelet."""
        return (pi ** -.25) * exp(-0.5 * (f - self.f0) ** 2.)

    def psi(self, t):
        """Morlet wavelet as described in Torrence and Compo (1998)"""
        return (pi ** -.25) * exp(1j * self.f0 * t - t ** 2. / 2.)

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)"""
        return (4 * pi) / (self.f0 + sqrt(2 + self.f0 ** 2))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)"""
        return 1. / sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time"""
        return 1. / coi

    def _set_f0(self, f0):
        # Sets the Morlet wave number, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.f0 = f0             # Wave number
        self.dofmin = 2          # Minimum degrees of freedom
        if self.f0 == 6.:
            self.cdelta = 0.776  # Reconstruction factor
            self.gamma = 2.32    # Decorrelation factor for time averaging
            self.deltaj0 = 0.60  # Factor for scale averaging
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1

class Paul:
    """Implements the Paul wavelet class.

    Note that the input parameter f is the angular frequency and that
    the default order for this wavelet is m=4.

    """

    name = 'Paul'

    def __init__(self, m=4):
        self._set_m(m)

    def psi_ft(self, f):
        """Fourier transform of the Paul wavelet."""
        return (2 ** self.m / sqrt(self.m * prod(range(2, 2 * self.m))) *
                f ** self.m * exp(-f) * (f > 0))

    def psi(self, t):
        """Paul wavelet as described in Torrence and Compo (1998)"""
        return (2 ** self.m * 1j ** self.m * prod(range(2, self.m - 1)) /
                sqrt(pi * prod(range(2, 2 * self.m + 1))) * (1 - 1j * t) **
                (-(self.m + 1)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)"""
        return 4 * pi / (2 * self.m + 1)

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)"""
        return sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time"""
        return 1. / coi

    def _set_m(self, m):
        # Sets the m derivative of a Gaussian, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.m = m               # Wavelet order
        self.dofmin =  2         # Minimum degrees of freedom
        if self.m == 4:
            self.cdelta = 1.132  # Reconstruction factor
            self.gamma = 1.17    # Decorrelation factor for time averaging
            self.deltaj0 = 1.50  # Factor for scale averaging
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1


class DOG:
    """Implements the derivative of a Guassian wavelet class.

    Note that the input parameter f is the angular frequency and that
    for m=2 the DOG becomes the Mexican hat wavelet, which is then
    default.

    """

    name = 'DOG'

    def __init__(self, m=2):
        self._set_m(m)

    def psi_ft(self, f):
        """Fourier transform of the DOG wavelet."""
        return (- 1j ** self.m / sqrt(gamma(self.m + 0.5)) * f ** self.m *
                exp(- 0.5 * f ** 2))

    def psi(self, t):
        """DOG wavelet as described in Torrence and Compo (1998)

        The derivative of a Gaussian of order n can be determined using
        the probabilistic Hermite polynomials. They are explicitly
        written as:
            Hn(x) = 2 ** (-n / s) * n! * sum ((-1) ** m) * (2 ** 0.5 *
                x) ** (n - 2 * m) / (m! * (n - 2*m)!)
        or in the recursive form:
            Hn(x) = x * Hn(x) - nHn-1(x)

        Source: http://www.ask.com/wiki/Hermite_polynomials

        """
        p = hermitenorm(self.m)
        return ((-1) ** (self.m + 1) * polyval(p, t) * exp(-t ** 2 / 2) /
                sqrt(gamma(self.m + 0.5)))

    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)"""
        return (2 * pi / sqrt(self.m + 0.5))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)"""
        return 1. / sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time"""
        return 1. / coi

    def _set_m(self, m):
        # Sets the m derivative of a Gaussian, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.m = m               # m-derivative
        self.dofmin = 1          # Minimum degrees of freedom
        if self.m == 2:
            self.cdelta = 3.541  # Reconstruction factor
            self.gamma = 1.43    # Decorrelation factor for time averaging
            self.deltaj0 = 1.40  # Factor for scale averaging
        elif self.m == 6:
            self.cdelta = 1.966
            self.gamma = 1.37
            self.deltaj0 = 0.97
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1


class Mexican_hat(DOG):
    """Implements the Mexican hat wavelet class.

    This class inherits the DOG class using m=2.

    """

    name = 'Mexican hat'

    def __init__(self):
        self._set_m(2)


def ar1(x):
    r"""Allen and Smith autoregressive lag-1 autocorrelation alpha. In a
    AR(1) model

        x(t) - <x> = \gamma(x(t-1) - <x>) + \alpha z(t) ,

    where <x> is the process mean, \gamma and \alpha are process
    parameters and z(t) is a Gaussian unit-variance white noise.

    PARAMETERS
        x (array like) :
            Univariate time series

    RETURNS
        g (float) :
            Estimate of the lag-one autocorrelation.
        a (float) :
            Estimate of the noise variance [var(x) ~= a**2/(1-g**2)]
        mu2 (foat) :
            Estimated square on the mean of a finite segment of AR(1)
            noise, mormalized by the process variance.

    REFERENCES
        [1] Allen, M. R. and Smith, L. A. (1996). Monte Carlo SSA:
            detecting irregular oscillations in the presence of colored
            noise. Journal of Climate, 9(12), 3373-3404.

    """
    x = asarray(x)
    N = x.size
    xm = x.mean()
    x = x - xm

    # Estimates the lag zero and one covariance
    c0 = x.transpose().dot(x) / N
    c1 = x[0:N-1].transpose().dot(x[1:N]) / (N - 1)

    # According to A. Grinsteds' substitutions
    B = -c1 * N - c0 * N**2 - 2 * c0 + 2 * c1 - c1 * N**2 + c0 * N
    A = c0 * N**2
    C = N * (c0 + c1 * N - c1)
    D = B**2 - 4 * A * C

    if D > 0:
        g = (-B - D**0.5) / (2 * A)
    else:
        raise Warning ('Cannot place an upperbound on the unbiased AR(1). '
            'Series is too short or trend is to large.')

    # According to Allen & Smith (1996), footnote 4
    mu2 = -1 / N + (2 / N**2) * ((N - g**N) / (1 - g) -
        g * (1 - g**(N - 1)) / (1 - g)**2)
    c0t = c0 / (1 - mu2)
    a = ((1 - g**2) * c0t) ** 0.5

    return g, a, mu2


def ar1_spectrum(freqs, ar1=0.) :
    """Lag-1 autoregressive theoretical power spectrum

    PARAMETERS
        ar1 (float) :
            Lag-1 autoregressive correlation coefficient.
        freqs (array like) :
            Frequencies at which to calculate the theoretical power
            spectrum.

    RETURNS
        Pk (array like) :
            Theoretical discrete Fourier power spectrum of noise signal.

    """
    # According to a post from the MadSci Network available at
    # http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html,
    # the time-series spectrum for an auto-regressive model can be
    # represented as
    #
    # P_k = \frac{E}{\left|1- \sum\limits_{k=1}^{K} a_k \, e^{2 i \pi
    #   \frac{k f}{f_s} } \right|^2}
    #
    # which for an AR1 model reduces to
    #
    freqs = asarray(freqs)
    Pk = (1 - ar1 ** 2) / abs(1 - ar1 * exp(-2 * pi * 1j * freqs)) ** 2

    # Theoretical discrete Fourier power spectrum of the noise signal following
    # Gilman et al. (1963) and Torrence and Compo (1998), equation 16.
    #N = len(freqs)
    #Pk = (1 - ar1 ** 2) / (1 + ar1 ** 2 - 2 * ar1 * cos(2 * pi * freqs / N))

    return Pk

def cwt(signal, dt=1., dj=1./12, s0=-1, J=-1, wavelet=Morlet(), result=None):
    """Continuous wavelet transform of the signal at specified scales.

    PARAMETERS
        signal (array like) :
            Input signal array
        dt (float) :
            Sample spacing.
        dj (float, optional) :
            Spacing between discrete scales. Default value is 0.25.
            Smaller values will result in better scale resolution, but
            slower calculation and plot.
        s0 (float, optional) :
            Smallest scale of the wavelet. Default value is 2*dt.
        J (float, optional) :
            Number of scales less one. Scales range from s0 up to
            s0 * 2**(J * dj), which gives a total of (J + 1) scales.
            Default is J = (log2(N*dt/so))/dj.
        wavelet (class, optional) :
            Mother wavelet class. Default is Morlet()
        result (string, optional) :
            If set to 'dictionary' returns the result arrays as itens
            of a dictionary.

    RETURNS
        W (array like) :
            Wavelet transform according to the selected mother wavelet.
            Has (J+1) x N dimensions.
        sj (array like) :
            Vector of scale indices given by sj = s0 * 2**(j * dj),
            j={0, 1, ..., J}.
        freqs (array like) :
            Vector of Fourier frequencies (in 1 / time units) that
            corresponds to the wavelet scales.
        coi (array like) :
            Returns the cone of influence, which is a vector of N
            points containing the maximum Fourier period of useful
            information at that particular time. Periods greater than
            those are subject to edge effects.
        fft (array like) :
            Normalized fast Fourier transform of the input signal.
        fft_freqs (array like):
            Fourier frequencies (in 1/time units) for the calculated
            FFT spectrum.

    EXAMPLE
        mother = wavelet.Morlet(6.)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
            0.25, 0.25, 0.5, 28, mother)

    """
    n0 = len(signal)                              # Original signal length.
    if s0 == -1: s0 = 2 * dt / wavelet.flambda()  # Smallest resolvable scale
    if J == -1: J = int(log2(n0 * dt / s0) / dj)  # Number of scales
    N = 2 ** (int(log2(n0)) + 1)                  # Next higher power of 2.
    signal_ft = fft(signal, N)                    # Signal Fourier transform
    ftfreqs = 2 * pi * fftfreq(N, dt)             # Fourier angular frequencies

    sj = s0 * 2. ** (arange(0, J+1) * dj)         # The scales
    freqs = 1. / (wavelet.flambda() * sj)         # As of Mallat 1999

    # Creates an empty wavlet transform matrix and fills it for every discrete
    # scale using the convolution theorem.
    W = zeros((len(sj), N), 'complex')
    for n, s in enumerate(sj):
        psi_ft_bar = ((s * ftfreqs[1] * N) ** .5 *
            conjugate(wavelet.psi_ft(s * ftfreqs)))
        W[n, :] = ifft(signal_ft * psi_ft_bar, N)

    # Checks for NaN in transform results and removes them from the scales,
    # frequencies and wavelet transform.
    sel = ~isnan(W).all(axis=1)
    sj = sj[sel]
    freqs = freqs[sel]
    W = W[sel, :]

    # Determines the cone-of-influence. Note that it is returned as a function
    # of time in Fourier periods. Uses triangualr Bartlett window with non-zero
    # end-points.
    coi = (n0 / 2. - abs(arange(0, n0) - (n0 - 1) / 2))
    coi = wavelet.flambda() * wavelet.coi() * dt * coi
    #
    if result == 'dictionary':
        result = dict(
            W = W[:, :n0],
            sj = sj,
            freqs = freqs,
            #period = 1. / freqs,
            coi = coi,
            signal_ft = signal_ft[1:N/2] / N ** 0.5,
            ftfreqs = ftfreqs[1:N/2] / (2. * pi),
            dt = dt,
            dj = dj,
            s0 = s0,
            J = J,
            wavelet = wavelet
        )
        return result
    else:
        return (W[:, :n0], sj, freqs, coi, signal_ft[1:N//2] / N ** 0.5,
                ftfreqs[1:N//2] / (2. * pi))

def icwt(W, sj, dt, dj=0.25, w=Morlet()):
    """Inverse continuous wavelet transform.

    PARAMETERS
        W (array like):
            Wavelet transform, the result of the cwt function.
        sj (array like):
            Vector of scale indices as returned by the cwt function.
        dt (float) :
            Sample spacing.
        dj (float, optional) :
            Spacing between discrete scales as used in the cwt
            function. Default value is 0.25.
        w (class, optional) :
            Mother wavelet class. Default is Morlet()

    RETURNS
        iW (array like) :
            Inverse wavelet transform.

    EXAMPLE
        mother = wavelet.Morlet(6.)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
            0.25, 0.25, 0.5, 28, mother)
        iwave = wavelet.icwt(wave, scales, 0.25, 0.25, mother)

    """
    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = ones([a, 1]) * sj
    else:
        raise Warning('Input array dimensions do not match.')

    # As of Torrence and Compo (1998), eq. (11)
    iW = dj * sqrt(dt) / w.cdelta * w.psi(0) * (real(W) / sj).sum(axis=0)
    return iW


def significance(signal, dt, scales, sigma_test=0, alpha=0.,
                 significance_level=0.95, dof=-1, wavelet=Morlet()):
    """
    Significance testing for the onde dimensional wavelet transform.

    PARAMETERS
        signal (array like or float) :
            Input signal array. If a float number is given, then the
            variance is assumed to have this value. If an array is
            given, then its variance is automatically computed.
        dt (float, optional) :
            Sample spacing. Default is 1.0.
        scales (array like) :
            Vector of scale indices given returned by cwt function.
        sigma_test (int, optional) :
            Sets the type of significance test to be performed.
            Accepted values are 0, 1 or 2. If omitted assume 0.

            If set to 0, performs a regular chi-square test, according
            to Torrence and Compo (1998) equation 18.

            If set to 1, performs a time-average test (equation 23). In
            this case, dof should be set to the number of local wavelet
            spectra that where averaged together. For the global
            wavelet spectra it would be dof=N, the number of points in
            the time-series.

            If set to 2, performs a scale-average test (equations 25 to
            28). In this case dof should be set to a two element vector
            [s1, s2], which gives the scale range that were averaged
            together. If, for example, the average between scales 2 and
            8 was taken, then dof=[2, 8].
        alpha (float, optional) :
            Lag-1 autocorrelation, used for the significance levels.
            Default is 0.0.
        significance_level (float, optional) :
            Significance level to use. Default is 0.95.
        dof (variant, optional) :
            Degrees of freedom for significance test to be set
            according to the type set in sigma_test.
        wavelet (class, optional) :
            Mother wavelet class. Default is Morlet().

    RETURNS
        signif (array like) :
            Significance levels as a function of scale.
        fft_theor (array like):
            Theoretical red-noise spectrum as a function of period.

    """
    try:
      n0 = len(signal)
    except:
      n0 = 1
    J = len(scales) - 1
    s0 = min(scales)
    dj = log2(scales[1] / scales[0])

    if n0 == 1:
      variance = signal
    else:
      variance = signal.std() ** 2

    period = scales * wavelet.flambda()  # Fourier equivalent periods
    freq = dt / period                   # Normalized frequency
    dofmin = wavelet.dofmin              # Degrees of freedom with no smoothing
    Cdelta = wavelet.cdelta              # Reconstruction factor
    gamma_fac = wavelet.gamma            # Time-decorrelation factor
    dj0 = wavelet.deltaj0                # Scale-decorrelation factor

    # Theoretical discrete Fourier power spectrum of the noise signal following
    # Gilman et al. (1963) and Torrence and Compo (1998), equation 16.
    pk = lambda k, a, N: (1 - a ** 2) / (1 + a ** 2 - 2 * a *
        cos(2 * pi * k / N))
    fft_theor = pk(freq, alpha, n0)
    fft_theor = variance * fft_theor     # Including time-series variance
    signif = fft_theor

    try:
        if dof == -1:
            dof = dofmin
    except:
        pass

    if sigma_test == 0:  # No smoothing, dof=dofmin, TC98 sec. 4
        dof = dofmin
        # As in Torrence and Compo (1998), equation 18
        chisquare = chi2.ppf(significance_level, dof) / dof
        signif = fft_theor * chisquare
    elif sigma_test == 1:  # Time-averaged significance
        if len(dof) == 1:
            dof = zeros(1, J+1) + dof
        sel = find(dof < 1)
        dof[sel] = 1
        # As in Torrence and Compo (1998), equation 23:
        dof = dofmin * (1 + (dof * dt / gamma_fac / scales) ** 2 ) ** 0.5
        sel = find(dof < dofmin)
        dof[sel] = dofmin  # Minimum dof is dofmin
        for n, d in enumerate(dof):
            chisquare = chi2.ppf(significance_level, d) / d;
            signif[n] = fft_theor[n] * chisquare
    elif sigma_test == 2:  # Time-averaged significance
        if len(dof) != 2:
            raise Exception('DOF must be set to [s1, s2], the range of scale-averages')
        if Cdelta == -1:
            raise Exception('Cdelta and dj0 not defined for %s with f0=%f' %
                             (wavelet.name, wavelet.f0))

        s1, s2 = dof
        sel = find((scales >= s1) & (scales <= s2));
        navg = sel.size
        if navg == 0:
            raise Exception('No valid scales between %d and %d.' % (s1, s2))

        # As in Torrence and Compo (1998), equation 25
        Savg = 1 / sum(1. / scales[sel])
        # Power-of-two mid point:
        Smid = exp((log(s1) + log(s2)) / 2.)
        # As in Torrence and Compo (1998), equation 28
        dof = (dofmin * navg * Savg / Smid) * ((1 + (navg * dj / dj0) ** 2) **
                                              0.5)
        # As in Torrence and Compo (1998), equation 27
        fft_theor = Savg * sum(fft_theor[sel] / scales[sel])
        chisquare = chi2.ppf(significance_level, dof) / dof;
        # As in Torrence and Compo (1998), equation 26
        signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare
    else:
        raise Exception('sigma_test must be either 0, 1, or 2.')

    return (signif, fft_theor)


def plotWavelet(signal, title, label, units, **kwargs):
    """
     Plot Wavelet Transfor for one signal

PARAMETER:
 signal : The signal that will be transformed
 title : Title of the plot
 label : Label
 units : unit of the data
 mother : The Mother  Wavelet. Default Morlet mother wavelet with wavenumber=6
 t0 : Initial time step
 dt : time step
 dj : Four sub-octaves per octaves
 s0 : Starting scale, here 6 months
 J : Seven powers of two with dj sub-octaves
 alpha: Lag-1 autocorrelation for white noise
 slevel : Significance level
 avg1,avg2 :  Range of periods to average
 nameSave : Path plus name to save the plot
 labelpowelog: set the x-axis in log scale
 showFig: Show Figure
 axQT: qt ax Figure
    """

    listParameters = ['mother', 't0', 'dt', 'dj', 's0', 'J', 'alpha',
                      'slevel', 'avg1', 'avg2', 'plotAv', 'fontsize',
                      'labelsize', 'labelpowelog','nameSave', 'showFig',
                      'xtickDate', 'axQT'
                      ]


    testeKeysArgs = [Ki for Ki in kwargs.keys() if Ki not in listParameters]

    if(len(testeKeysArgs) >= 1):
        print('The following keys args are not defined: ', testeKeysArgs)
        return


    if 'mother' in kwargs.keys():
        mother = kwargs['kwarags']

    else:
        mother = Morlet(6.)

    if't0' in kwargs.keys():
        t0 = kwargs['t0']

    else:
        t0=1.0

    if'dt' in kwargs.keys():
        dt = kwargs['dt']

    else:
        dt=1.0

    if'dj' in kwargs.keys():
        dj = kwargs['dj']
    else:
        dj=0.25

    if 's0'in kwargs.keys():
        s0 = kwargs['s0']
    else:
        s0=-1

    if 'J' in kwargs.keys():
        J = kwargs['J']
    else:
        J = -1

    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 0.0

    if 'slevel' in kwargs.keys():
        slevel = kwargs['slevel']
    else:
        slevel = 0.95

    if 'avg1' in kwargs.keys():
        avg1 = kwargs['avg1']
    else:
        avg1 = 15

    if 'avg2' in kwargs.keys():
        avg1 = kwargs['avg2']
    else:
        avg2 =20

    if 'plotAv' in kwargs.keys():
        plotAv = kwargs['plotAv']
    else:
        plotAv = 0

    if 'nameSave' in kwargs.keys():
        nameSave = kwargs['nameSave']
    else:
        nameSave = None

    if 'fontsize' in kwargs.keys():
        fontsize = kwargs['fontsize']
    else:
        fontsize = 15

    if 'labelsize' in kwargs.keys():
        labelsize = kwargs['labelsize']
    else:
        labelsize = 18

    if 'labelpowelog' in kwargs.keys():
        labelpowelog = kwargs['labelpowelog']
    else:
        labelpowelog = False

    if 'showFig' in kwargs.keys():
        showFig = kwargs['showFig']
    else:
        showFig = True
    if('xtickDate' in kwargs.keys()):
        xtickDate = kwargs['xtickDate']
    else:
        xtickDate = None

    if('axQT' in kwargs.keys()):
        axQT = kwargs['axQT']
    else:
        axQT = None


    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['axes.labelsize'] = labelsize

    var = signal


    std = var.std()                      # Standard deviation
    std2 = std ** 2                      # Variance
    var = (var - var.mean()) / std       # Calculating anomaly and normalizing

    N = var.size                         # Number of measurements

    if(xtickDate is not None):
        time = mdates.date2num(xtickDate)
    else:
        time = numpy.arange(0, N) * dt + t0  # Time array in years

    dj = 0.25                            # Four sub-octaves per octaves
    s0 = -1 #2 * dt                      # Starting scale, here 6 months
    J = -1 # 7 / dj                      # Seven powers of two with dj sub-octaves
    alpha = 0.0                          # Lag-1 autocorrelation for white noise
    wave, scales, freqs, coi, fft, fftfreqs = cwt(var, dt, dj, s0, J,mother)
    iwave = icwt(wave, scales, dt, dj, mother)
    power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
    fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
    period = 1. / freqs

    signif, fft_theor = significance(1.0, dt, scales, 0, alpha,
                            significance_level=slevel, wavelet=mother)
    sig95 = (signif * numpy.ones((N, 1))).transpose()
    sig95 = power / sig95                # Where ratio > 1, power is significant

    # Calculates the global wavelet spectrum and determines its significance level.
    glbl_power = std2 * power.mean(axis=1)
    dof = N - scales                     # Correction for padding at edges
    glbl_signif, tmp = significance(std2, dt, scales, 1, alpha,
                           significance_level=slevel, dof=dof, wavelet=mother)

    # Scale average between avg1 and avg2 periods and significance level
    sel = find((period >= avg1) & (period < avg2))
    Cdelta = mother.cdelta
    scale_avg = (scales * numpy.ones((N, 1))).transpose()
    # As in Torrence and Compo (1998) equation 24
    scale_avg = power / scale_avg
    scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = significance(std2, dt, scales, 2, alpha,
                                significance_level=slevel, dof=[scales[sel[0]],
                                scales[sel[-1]]], wavelet=mother)

    # The following routines plot the results in four different subplots containing
    # the original series anomaly, the wavelet power spectrum, the global wavelet
    # and Fourier spectra and finally the range averaged wavelet spectrum. In all
    # sub-plots the significance levels are either included as dotted lines or as
    # filled contour lines.
    #pylab.close('all')
    fontsize = 'medium'
    params = {'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize
             }
    pylab.rcParams.update(params)          # Plot parameters


    figprops = dict(figsize=(11, 8), dpi=72)
    fig = pylab.figure(**figprops)


    # First sub-plot, the original time series anomaly.
    ax = pylab.axes([0.1, 0.75, 0.65, 0.2])
    #ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax.plot(time, var, 'k', linewidth=1.5)
    ax.set_title('a) %s' % (title, ))
    if(xtickDate is not None):
        ax.xaxis_date()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    if units != '':
        ax.set_ylabel(r'%s [$%s$/$%s$]' % (label, units, units, ))
    else:
      ax.set_ylabel(r'%s' % (label, ))

    #Bug...
    #BAD SMELL.
    #Why is zero the last term of the coi
    # when the signal is filtered?
    #... Bug.***:***
    tmpI = numpy.where(coi == 0)
    if(len(tmpI[0]) != 0):
        if(tmpI[0] == len(coi) -1):
            coi[tmpI] = 0.1 * coi[-2]

    # Second sub-plot, the normalized wavelet power spectrum and significance level
    # contour lines and cone of influece hatched area.
    bx = pylab.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bx.contourf(time, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
                extend='both')
    bx.contour(time, numpy.log2(period), sig95, [-99, 1], colors='k',
               linewidths=2.)
    bx.fill(numpy.concatenate([time[:1]-dt, time, time[-1:]+dt, time[-1:]+dt,
            time[:1]-dt, time[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi,
            [1e-9], period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3,
            hatch='x')
    bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
    bx.set_ylabel('Period (Days)')
    Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                               numpy.ceil(numpy.log2(period.max())))
    bx.set_yticks(numpy.log2(Yticks))
    bx.set_yticklabels(Yticks)
    bx.invert_yaxis()
    if(xtickDate is not None):
        bx.xaxis_date()
        bx.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    if(axQT is not None):
        axQT = bx

    # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
    # noise spectra.
    cx = pylab.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
    cx.plot(glbl_signif, numpy.log2(period), 'k--')
    cx.plot(fft_power, numpy.log2(1./fftfreqs), '-', color=[0.7, 0.7, 0.7],
            linewidth=1.)
    cx.plot(glbl_power, numpy.log2(period), 'k-', linewidth=1.5)
    cx.set_title('c) Global Wavelet Spectrum')
    if units != '':
        cx.set_xlabel(r'Power [$%s^2$]' % (units, ))
    else:
        cx.set_xlabel(r'Power')

    if(labelpowelog):
        cx.set_xscale('log')
    #cx.set_xlim([0, glbl_power.max() + std2])
    cx.set_ylim(numpy.log2([period.min(), period.max()]))
    cx.set_yticks(numpy.log2(Yticks))
    cx.set_yticklabels(Yticks)
    pylab.setp(cx.get_yticklabels(), visible=False)
    cx.invert_yaxis()

    if(plotAv == 1):

        # Fourth sub-plot, the scale averaged wavelet spectrum as
        #determined by the
        # avg1 and avg2 parameters
        dx = pylab.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
        dx.plot(time, scale_avg, 'k-', linewidth=1.5)
        dx.set_title('d) $%d$-$%d$ year scale-averaged power' % (avg1, avg2))
        dx.set_xlabel('Time (days)')
        if units != '':
            dx.set_ylabel(r'Average variance [$%s$]' % (units, ))
        else:
            dx.set_ylabel(r'Average variance')
    else:
        bx.set_xlabel('Time (days)')

    #
    ax.set_xlim([time.min(), time.max()])
    #
    pylab.draw()
    if(nameSave):
        pylab.savefig(nameSave)

    if(showFig):
        pylab.show()

    return fig

def rect(x, normalize=False) :
    if type(x) in [int, float]:
        shape = [x, ]
    elif type(x) in [list, dict]:
        shape = x
    elif type(x) in [numpy.ndarray, numpy.ma.core.MaskedArray]:
        shape = x.shape
    X = zeros(shape)
    X[0] = X[-1] = 0.5
    X[1:-1] = 1

    if normalize:
        X /= X.sum()

    return X
############################################################################
############################################################################
############################################################################
############################################################################

class wcoherence:
    """
    This class is an Python interface for the Wavelet Coherence matlab functions of the package for wavelet,
    cross-wavelet and coherence-wavelet analises profided by
    Aslak Grinsted, John C. Moore and Svetlana Jevrejeva.

    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence

    However, the Continuous wavelet transform of the signal, in this class, is a pure python
    function.
    Callable Class:
    RETURN
        Rsq : Coherence Wavelet
        period: a vector of "Fourier" periods associated with Wxy
        scale: a vector of wavelet scales associated with Wxy
        coi: the cone of influence
        sig95: Significance
    """
    def __init__(self, signal1, signal2):
        HOME = os.path.expanduser('~')
        mFiles = HOME+'/.piwavelet/wtc/'
        self.wtcPath = octave.addpath(mFiles)
        self.signal1 = signal1
        self.signal2 = signal2
        self.Rqs, self.period, self.scale, self.coi, self.wtcsig = self.__wtc(self.signal1, self.signal2)
        self.freqs = 1.0/self.period

    def __call__(self):
        return self.Rqs, self.period, self.scale, self.coi, self.wtcsig


    def __wtc(self, signal1, signal2):
        """
        Wavelet Coherence

USAGE:
    Rsq,period,scale,coi,sig95=wtc(x,y)

RETURN:
    Rsq : Coherence Wavelet
    period: a vector of "Fourier" periods associated with Wxy
    scale: a vector of wavelet scales associated with Wxy
    coi: the cone of influence
    sig95: Significance



        """
        Rqs, period,scale,coi,wtcsig = octave.wtc(signal1, signal2, nout=5)
        period =period[0]
        scale = scale[0]
        coi = coi[0]
        return Rqs, period,scale,coi,wtcsig

    def plot(self, t, title, units,   **kwargs ):
        """
        Plots the wavelet coherence

            PARAMETERS
                title: Title of the Plot
                units: (string) Units of the period and time  (e.g. 'days')
                t : array with time
                OPTIONALS:
                    gray : (boolean) True for gray map .
                    levels : List with significance level that will be showed in the plot
                    labels : List with the Label of significance level that will be apper into the color bar. If not defined, the levels list is used instead
                    pArrow : (boolean)  True for draw vector of phase angle (it has problem not recomended for large sample of data)
                    pSigma : (boolean) True for draw the significance countor lines
                    nameSave : (string) path plus name to save the figure, if it is define, the plot is saved but not showed
                    scale : (boolean) True  for not log2 scale of the Plot
        """


        self.__plotWC(self.Rqs, t, self.coi, self.freqs, self.wtcsig, title,\
                               units, **kwargs)



    def __plotWC(self,  wc, t, coi, freqs, signif, title, units='days', **kwargs):
        """Plots the wavelet coherence

        PARAMETERS
            wc (array like) :
                 Coherence Wavelet
            coi (array like) :
                Cone of influence, which is a vector of N points containing
                the maximum Fourier period of useful information at that
                particular time. Periods greater than those are subject to
                edge effects.
            freqs (array like) :
                Vector of Fourier equivalent frequencies (in 1 / time units)
                that correspond to the wavelet scales.
            signif (array like) :
                Significance levels as a function of Fourier equivalent
                frequencies.
            OPTIONALS:
                    gray : (boolean) True for gray map .
                    levels : List with significance level that will be showed in the plot
                    labels : List with the Label of significance level that will be apper into the color bar. If not defined, the levels list is used instead
                    pArrow : (boolean)  True for draw vector of phase angle (it has problem not recomended for large sample of data)
                    pSigma : (boolean) True for draw the significance countor lines
                    nameSave : (string) path plus name to save the figure, if it is define, the plot is saved but not showed
                    scale : (boolean) True  for not log2 scale of the Plot


        RETURNS
            A list with the figure and axis objects for the plot.



        """

        listParameters = ['levels',  'labels',  'pArrow',  'pSigma',  'gray',
                          'nameSave',  'scale', 'zoom', 'labelsize']


        testeKeysArgs = [Ki for Ki in kwargs.keys() if Ki not in  listParameters]

        if(len(testeKeysArgs) >=1):
            print('The following key args are not defined: ',  testeKeysArgs)
            return

        # Sets some parameters and renames some of the input variables.
        from matplotlib import pyplot

        if 'levels' in kwargs.keys():
            levels = kwargs['levels']
        else:
            levels=None

        if 'labels' in kwargs.keys():
            levels = kwargs['labels']
        else:
            labels=None

        if 'pArrow' in kwargs.keys():
            levels = kwargs['pArrow']
        else:
            pArrow=None

        if 'pSigma' in kwargs.keys():
            levels = kwargs['pSigma']
        else:
            pSigma=True

        if 'gray' in kwargs.keys():
            levels = kwargs['gray']
        else:
            gray = None

        if 'nameSave' in kwargs.keys():
            levels = kwargs['nameSave']
        else:
            nameSave = None

        if 'scale' in kwargs.keys():
            scale = kwargs['scale']
        else:
            scale = 'log2'


        if 'zoom' in kwargs.keys():
            if(len(kwargs['zoom'])<=1 or len(kwargs['zoom']) >2):
                zoom = None
            else:
                zoom = kwargs['zoom']
        else:
            zoom = None

        if('labelsize' in kwargs.keys()):
            labelsize = kwargs['labelsize']
            labelsize = int(labelsize)

        else:
            labelsize = 18


        fontsize = 'medium'
        params = {'font.family': 'serif',
                          'font.sans-serif': ['Helvetica'],
                          'font.size': 25,
                          'font.stretch': 'ultra-condensed',
                          'xtick.labelsize': labelsize,
                          'ytick.labelsize': labelsize,
                          'axes.titlesize': fontsize,
                          'timezone': 'UTC'
                         }
        pyplot.rcParams.update(params)
        pyplot.ion()
        fp=dict()
        ap=dict(left=0.15, bottom=0.12, right=0.95, top=0.95,\
                     wspace=0.10, hspace=0.10)
        orientation='landscape'
        fig = pyplot.figure(**fp)
        fig.subplots_adjust(**ap)


        N = len(t)
        dt = t[1] - t[0]
        period = 1. / freqs
        power = wc
#        sig95 = numpy.ones([1, N]) * signif[:, None]
#        print signif.shape
#        raw_input('oi')
        sig95 = signif # power is significant where ratio > 1


        # Calculates the phase between both time series. The phase arrows in the
        # cross wavelet power spectrum rotate clockwise with 'north' origin.

        angle = 0.5 * numpy.pi - numpy.angle(wc)
        u, v = numpy.cos(angle), numpy.sin(angle)

        result = []


        da = [3, 3]

        fig = fig
        result.append(fig)


        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('%s' %title)
        ax.set_xlabel('Time (%s)' %units)
        ax.set_ylabel('Period (%s)' %units)


        # Plots the cross wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.

        if(levels):
            if(labels):
                pass
            else:
                labels = [str(li) for li in levels]
        else:
            levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9, 1]
            labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7',  '0.8', '0.9','1']
        cmin, cmax = power.min(), power.max()
        rmin, rmax = min(levels), max(levels)

        if (cmin < rmin) & (cmax > rmax):
            extend = 'both'
        elif (cmin < rmin) & (cmax <= rmax):
            extend = 'min'
        elif (cmin >= rmin) & (cmax > rmax):
            extend = 'max'
        elif (cmin >= rmin) & (cmax <= rmax):
            extend = 'neither'

        if scale == 'log2':
            Power = numpy.log2(power)
            Levels = numpy.log2(levels)
        else:
            Power = power
            Levels = levels

        norml =colors.BoundaryNorm(Levels, 256)

        if(gray == True):
            cf = ax.contourf(t, numpy.log2(period), Power, Levels, cmap = plt.cm.gray, norm=norml, extend=extend)
        else:
            cf = ax.contourf(t, numpy.log2(period), Power, Levels, cmap = plt.cm.jet, norm=norml, extend=extend)

        if(pSigma):
            ax.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k',
                linewidths=2.)

        if( pArrow):
            q = ax.quiver(t[::da[1]], numpy.log2(period)[::da[0]], u[::da[0], ::da[1]],
                v[::da[0], ::da[1]], units='width', angles='uv', pivot='mid',
                linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
                headaxislength=5, minshaft=2, minlength=5)

        if(zoom):
            newPeriod = period[find((period>=zoom[0])&(period<=zoom[1]))]
            ax.fill(numpy.concatenate([t[:1]-dt, t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt,
                t[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi, [1e-9],
                period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3, hatch='x')
            Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                numpy.ceil(numpy.log2(period.max())))
            ax.set_yticks(numpy.log2(Yticks))
            ax.set_yticklabels(Yticks)
            ax.set_xlim([t.min(), t.max()])
            ax.set_ylim(numpy.log2([newPeriod.min(), min([coi.max(), newPeriod.max()])]))
            ax.invert_yaxis()
            cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
            cbar.ax.set_yticklabels(labels)

            pylab.draw()

            if(nameSave):
                pylab.savefig(nameSave)
            else:
                pylab.show()
        else:

            ax.fill(numpy.concatenate([t[:1]-dt, t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt,
                t[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi, [1e-9],
                period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3, hatch='x')
            Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                numpy.ceil(numpy.log2(period.max())))
            ax.set_yticks(numpy.log2(Yticks))
            ax.set_yticklabels(Yticks)
            ax.set_xlim([t.min(), t.max()])
            ax.set_ylim(numpy.log2([period.min(), min([coi.max(), period.max()])]))
            ax.invert_yaxis()
            cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
            cbar.ax.set_yticklabels(labels)

            pylab.draw()

            if nameSave is not None:
                pylab.savefig(nameSave)
            else:
                pylab.show()


        result.append(ax)

        return result

#*********************************************************************************************************
# Cross wavelet  Spectrun
#*********************************************************************************************************

class wcross:
    """
    This class is an Python interface for the Cross wavelet Spectrun matlab functions of the package for wavelet,
    cross-wavelet and coherence-wavelet analises profided by
    Aslak Grinsted, John C. Moore and Svetlana Jevrejeva.

    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence

    However, the Continuous wavelet transform of the signal, in this class, is a pure python
    function.
    """
    def __init__(self, signal1, signal2):
        HOME = os.path.expanduser('~')
        mFiles = HOME+'/.piwavelet/wtc/'
        self.wtcPath = octave.addpath(mFiles)
        self.signal1 = signal1
        self.signal2 = signal2
        self.xwt, self.period, self.scale, self.coi, self.signif = self.__xtc(self.signal1, self.signal2)
        self.freqs = 1.0/self.period

    def __call__(self):
        return self.xwt, self.period, self.scale, self.coi, self.signif


    def __xtc(self, signal1, signal2):
        """Cross wavelet transform

USAGE:
    Wxy,period,scale,coi,sig95=xwt(x,y)

RETURN
    Wxy: the cross wavelet transform of x against y
    period: a vector of "Fourier" periods associated with Wxy
    scale: a vector of wavelet scales associated with Wxy
    coi: the cone of influence
    sig95: Significance
        """
        xwt, period,scale,coi,signif =octave.xwt(signal1, signal2, nout=5)
        period =period[0]
        scale = scale[0]
        coi = coi[0]

        return xwt, period,scale,coi,signif

    def plot(self, t, title, units, **kwargs ):
        """
        Plots the wavelet coherence

            PARAMETERS
                title: Title of the Plot
                units: (string) Units of the period and time  (e.g. 'days')
                t : array with time
                OPTIONALS:
                    gray : (boolean) True for gray map .
                    levels : List with significance level that will be showed in the plot
                    labels : List with the Label of significance level that will be apper into the color bar. If not defined, the levels list is used instead
                    pArrow : (boolean)  True for draw vector of phase angle (it has problem not recomended for large sample of data)
                    pSigma : (boolean) True for draw the significance countor lines
                    nameSave : (string) path plus name to save the figure, if it is define, the plot is saved but not showed
                    scale : (boolean) True  for not log2 scale of the Plot
        """
        self.__plotXWC(self.xwt, t, self.coi, self.freqs, self.signif, title, units,**kwargs)

    def __plotXWC(self,  xwt, t, coi, freqs, signif, title, units='days',**kwargs):
        """Plots the cross-wavelet power spectrun

        PARAMETERS
            xwt (array like) :
                Cross wavelet transform.
            coi (array like) :
                Cone of influence, which is a vector of N points containing
                the maximum Fourier period of useful information at that
                particular time. Periods greater than those are subject to
                edge effects.
            freqs (array like) :
                Vector of Fourier equivalent frequencies (in 1 / time units)
                that correspond to the wavelet scales.
            signif (array like) :
                Significance levels as a function of Fourier equivalent
                frequencies.
            OPTIONALS:
                    gray : (boolean) True for gray map .
                    levels : List with significance level that will be showed in the plot
                    labels : List with the Label of significance level that will be apper into the color bar. If not defined, the levels list is used instead
                    pArrow : (boolean)  True for draw vector of phase angle (it has problem not recomended for large sample of data)
                    pSigma : (boolean) True for draw the significance countor lines
                    nameSave : (string) path plus name to save the figure, if it is define, the plot is saved but not showed
                    scale : (boolean) True  for not log2 scale of the Plot


        RETURNS
            A list with the figure and axis objects for the plot.



        """
        # Sets some parameters and renames some of the input variables.

        listParameters = ['levels',  'labels',  'pArrow',  'pSigma',  'gray',  'nameSave',  'scale', 'zoom']


        testeKeysArgs = [Ki for Ki in kwargs.keys() if Ki not in  listParameters]

        if(len(testeKeysArgs) >=1):
            print('The following key args are not defined: ',  testeKeysArgs)
            return

        from matplotlib import pyplot

        if 'levels' in kwargs.keys():
            levels = kwargs['levels']
        else:
            levels=None

        if 'labels' in kwargs.keys():
            levels = kwargs['labels']
        else:
            labels=None

        if 'pArrow' in kwargs.keys():
            levels = kwargs['pArrow']
        else:
            pArrow=None

        if 'pSigma' in kwargs.keys():
            levels = kwargs['pSigma']
        else:
            pSigma=True

        if 'gray' in kwargs.keys():
            levels = kwargs['gray']
        else:
            gray = None

        if 'nameSave' in kwargs.keys():
            levels = kwargs['nameSave']
        else:
            nameSave = None

        if 'scale' in kwargs.keys():
            scale = kwargs['scale']
        else:
            scale = 'log2'


        fontsize = 'medium'
        params = {'font.family': 'serif',
                          'font.sans-serif': ['Helvetica'],
                          'font.size': 18,
                          'font.stretch': 'ultra-condensed',
                          'xtick.labelsize': fontsize,
                          'ytick.labelsize': fontsize,
                          'axes.titlesize': fontsize,
                          'timezone': 'UTC'
                         }
        pyplot.rcParams.update(params)
        pyplot.ion()
        fp=dict()
        ap=dict(left=0.15, bottom=0.12, right=0.95, top=0.95,\
                     wspace=0.10, hspace=0.10)
        orientation='landscape'
        fig = pyplot.figure(**fp)
        fig.subplots_adjust(**ap)





        N = len(t)
        dt = t[1] - t[0]
        period = 1. / freqs
        power = abs(xwt)
#        sig95 = numpy.ones([1, N]) * signif[:, None]
#        print signif.shape
#        raw_input('oi')
        sig95 = signif # power is significant where ratio > 1


        # Calculates the phase between both time series. The phase arrows in the
        # cross wavelet power spectrum rotate clockwise with 'north' origin.

        angle = numpy.angle(xwt) #+ 0.5 * numpy.pi
        u, v = numpy.cos(angle), numpy.sin(angle)

        result = []


        da = [3, 3]

        fig = fig
        result.append(fig)


        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('%s' %title)
        ax.set_xlabel('Time (%s)' %units)
        ax.set_ylabel('Period (%s)' %units)


        # Plots the cross wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.

        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']
        cmin, cmax = power.min(), power.max()
        rmin, rmax = min(levels), max(levels)

        if (cmin < rmin) & (cmax > rmax):
            extend = 'both'
        elif (cmin < rmin) & (cmax <= rmax):
            extend = 'min'
        elif (cmin >= rmin) & (cmax > rmax):
            extend = 'max'
        elif (cmin >= rmin) & (cmax <= rmax):
            extend = 'neither'

        if scale == 'log2':
            Power = numpy.log2(power)
            Levels = numpy.log2(levels)
        else:
            Power = power
            Levels = levels

        norml = colors.BoundaryNorm(Levels, 256)
        if(gray == True):
            cf = ax.contourf(t, numpy.log2(period), Power, Levels, cmap = plt.cm.gray, norm=norml, extend=extend)
        else:
            cf = ax.contourf(t, numpy.log2(period), Power, Levels, cmap = plt.cm.jet, norm=norml, extend=extend)

        if(pSigma):
            ax.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k',
                linewidths=2.)


        if(pArrow):
            q = ax.quiver(t[::da[1]], numpy.log2(period)[::da[0]], u[::da[0], ::da[1]],
                v[::da[0], ::da[1]],
                units='width', angles='uv', pivot='mid'
                #linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
                #headaxislength=5, minshaft=2, minlength=5
                )


        ax.fill(numpy.concatenate([t[:1]-dt, t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt,
            t[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi, [1e-9],
            period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3, hatch='x')
        Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
            numpy.ceil(numpy.log2(period.max())))
        ax.set_yticks(numpy.log2(Yticks))
        ax.set_yticklabels(Yticks)
        ax.set_xlim([t.min(), t.max()])
        ax.set_ylim(numpy.log2([period.min(), min([coi.max(), period.max()])]))
        ax.invert_yaxis()
        cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
        cbar.ax.set_yticklabels(labels)

        pylab.draw()

        if(nameSave):
            pylab.savefig(nameSave)
        else:
            pylab.show()
        result.append(ax)

        return result

class smooth:
    """
    This class is an Python interface for the Smoothing matlab functions of the package for wavelet,
    cross-wavelet and coherence-wavelet analises profided by
    Aslak Grinsted, John C. Moore and Svetlana Jevrejeva.

    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence

    However, the Continuous wavelet transform of the signal, in this class, is a pure python
    function.

    Smoothing as in the appendix of Torrence and Webster "Inter decadal changes in the ENSO-Monsoon System" 1998
    used in wavelet coherence calculations. Only applicable for the Morlet wavelet.
    """
    def __init__(self, wave,dt,period,dj,scale):
        HOME = os.path.expanduser('~')
        mFiles = HOME+'/.piwavelet/wtc/'
        self.wtcPath = octave.addpath(mFiles)
        self.wave,self.dt,self.period,self.dj,self.scale = wave,dt,period,dj,scale


    def __call__(self):

        return self.__smoothwavelet(self.wave,self.dt,self.period,self.dj,self.scale)

    def __smoothwavelet(self, wave,dt,period,dj,scale):
        """
        Smoothing as in the appendix of Torrence and Webster "Inter decadal changes in the ENSO-Monsoon System" 1998
        used in wavelet coherence calculations. Only applicable for the Morlet wavelet.
        """

        swave = octave.call('smoothwavelet', wave,dt,period,dj,scale)
        return swave

    def __smooth(self, W, dt, dj, scales):
        """Smoothing function used in coherence analysis."""
        # The smoothing is performed by using a filter given by the absolute
        # value of the wavelet function at each scale, normalized to have a
        # total weight of unity, according to suggestions by Torrence &
        # Webster (1999) and bry Grinsted et al. (2004).

        m, n = W.shape
        T = zeros([m, n])
        T = matrix(T)
        W = matrix(W)

        # Filter in time.
        npad = int(2 ** ceil(log2(n)))
        N = npad - n
        k = 2 * pi * fftfreq(npad-N)
        k2 = k ** 2
        snorm = scales / dt

        for i in range(m):
            F = exp(-0.5 * (snorm[i] ** 2) * k2)
            smooth = ifft(F * fft(W[i, :], npad-N))
            print(smooth.shape, T[i, :].shape,F.shape, npad)
            T[i, :] = smooth[0:n]

        if isreal(W).all():
            T = T.real

        # Filter in scale. For the Morlet wavelet it's simply a boxcar with
        # 0.6 width.
        wsize = 0.6/ dj * 2
        win = rect(int(round(wsize)), normalize=True)
        T = convolve2d(T, win[:, None], 'same')

        return T

##*********************************************************************************************************
##  Continuous wavelet transform
##*********************************************************************************************************
#
#class waveletCC:
#    """
#    Continuous wavelet transform of the signal at specified scales.
#
#    """
#
#    def __init__(self):
#        pass
#
#    def cwt(self, signal, dt, dj=0.25, s0=-1, J=-1, wavelet=Morlet()):
#        """Continuous wavelet transform of the signal at specified scales.
#
#        PARAMETERS
#            signal (array like) :
#                Input signal array
#            dt (float) :
#                Sample spacing.
#            dj (float, optional) :
#                Spacing between discrete scales. Default value is 0.25.
#                Smaller values will result in better scale resolution, but
#                slower calculation and plot.
#            s0 (float, optional) :
#                Smallest scale of the wavelet. Default value is 2*dt.
#            J (float, optional) :
#                Number of scales less one. Scales range from s0 up to
#                s0 * 2**(J * dj), which gives a total of (J + 1) scales.
#                Default is J = (log2(N*dt/so))/dj.
#            wavelet (class, optional) :
#                Mother wavelet class. Default is Morlet()
#
#        RETURNS
#            W (array like) :
#                Wavelet transform according to the selected mother wavelet.
#                Has (J+1) x N dimensions.
#            sj (array like) :
#                Vector of scale indices given by sj = s0 * 2**(j * dj),
#                j={0, 1, ..., J}.
#            freqs (array like) :
#                Vector of Fourier frequencies (in 1 / time units) that
#                corresponds to the wavelet scales.
#            coi (array like) :
#                Returns the cone of influence, which is a vector of N
#                points containing the maximum Fourier period of useful
#                information at that particular time. Periods greater than
#                those are subject to edge effects.
#            fft (array like) :
#                Normalized fast Fourier transform of the input signal.
#            fft_freqs (array like):
#                Fourier frequencies (in 1/time units) for the calculated
#                FFT spectrum.
#
#        EXAMPLE
#            mother = wavelet.Morlet(6.)
#            wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
#                0.25, 0.25, 0.5, 28, mother)
#
#        """
#        n0 = len(signal)                              # Original signal length.
#        if s0 == -1: s0 = 2 * dt / wavelet.flambda()  # Smallest resolvable scale
#        if J == -1: J = int(log2(n0 * dt / s0) / dj)  # Number of scales
#        N = 2 ** (int(log2(n0)) + 1)                  # Next higher power of 2.
#        signal_ft = fft(signal, N)                    # Signal Fourier transform
#        ftfreqs = 2 * pi * fftfreq(N, dt)             # Fourier angular frequencies
#
#        sj = s0 * 2. ** (arange(0, J+1) * dj)         # The scales
#        freqs = 1. / (wavelet.flambda() * sj)         # As of Mallat 1999
#
#        # Creates an empty wavlet transform matrix and fills it for every discrete
#        # scale using the convolution theorem.
#        W = zeros((len(sj), N), 'complex')
#        for n, s in enumerate(sj):
#            psi_ft_bar = (s * ftfreqs[1] * N) ** .5 * conjugate(wavelet.psi_ft(s * ftfreqs))
#            W[n, :] = ifft(signal_ft * psi_ft_bar, N)
#
#        # Checks for NaN in transform results and removes them from the scales,
#        # frequencies and wavelet transform.
#        sel = ~isnan(W).all(axis=1)
#        sj = sj[sel]
#        freqs = freqs[sel]
#        W = W[sel, :]
#
#        # Determines the cone-of-influence. Note that it is returned as a function
#        # of time in Fourier periods. Uses triangualr Bartlett window with non-zero
#        # end-points.
#        coi = (n0 / 2. - abs(arange(0, n0) - (n0 - 1) / 2))
#        coi = wavelet.flambda() * wavelet.coi() * dt * coi
#        #
#        return (W[:, :n0], sj, freqs, coi, signal_ft[1:N/2] / N ** 0.5,
#                ftfreqs[1:N/2] / (2. * pi))
#
#    def significance(self, signal, dt, scales, sigma_test=0, alpha=0.,
#                 significance_level=0.95, dof=-1, wavelet=Morlet()):
#        """
#        Significance testing for the onde dimensional wavelet transform.
#
#        PARAMETERS
#            signal (array like or float) :
#                Input signal array. If a float number is given, then the
#                variance is assumed to have this value. If an array is
#                given, then its variance is automatically computed.
#            dt (float, optional) :
#                Sample spacing. Default is 1.0.
#            scales (array like) :
#                Vector of scale indices given returned by cwt function.
#            sigma_test (int, optional) :
#                Sets the type of significance test to be performed.
#                Accepted values are 0, 1 or 2. If omitted assume 0.
#
#                If set to 0, performs a regular chi-square test, according
#                to Torrence and Compo (1998) equation 18.
#
#                If set to 1, performs a time-average test (equation 23). In
#                this case, dof should be set to the number of local wavelet
#                spectra that where averaged together. For the global
#                wavelet spectra it would be dof=N, the number of points in
#                the time-series.
#
#                If set to 2, performs a scale-average test (equations 25 to
#                28). In this case dof should be set to a two element vector
#                [s1, s2], which gives the scale range that were averaged
#                together. If, for example, the average between scales 2 and
#                8 was taken, then dof=[2, 8].
#            alpha (float, optional) :
#                Lag-1 autocorrelation, used for the significance levels.
#                Default is 0.0.
#            significance_level (float, optional) :
#                Significance level to use. Default is 0.95.
#            dof (variant, optional) :
#                Degrees of freedom for significance test to be set
#                according to the type set in sigma_test.
#            wavelet (class, optional) :
#                Mother wavelet class. Default is Morlet().
#
#        RETURNS
#            signif (array like) :
#                Significance levels as a function of scale.
#            fft_theor (array like):
#                Theoretical red-noise spectrum as a function of period.
#
#        """
#        try:
#          n0 = len(signal)
#        except:
#          n0 = 1
#        J = len(scales) - 1
#        s0 = min(scales)
#        dj = log2(scales[1] / scales[0])
#
#        if n0 == 1:
#          variance = signal
#        else:
#          variance = signal.std() ** 2
#
#        period = scales * wavelet.flambda()  # Fourier equivalent periods
#        freq = dt / period                   # Normalized frequency
#        dofmin = wavelet.dofmin              # Degrees of freedom with no smoothing
#        Cdelta = wavelet.cdelta              # Reconstruction factor
#        gamma_fac = wavelet.gamma            # Time-decorrelation factor
#        dj0 = wavelet.deltaj0                # Scale-decorrelation factor
#
#        # Theoretical discrete Fourier power spectrum of the noise signal following
#        # Gilman et al. (1963) and Torrence and Compo (1998), equation 16.
#        pk = lambda k, a, N: (1 - a ** 2) / (1 + a ** 2 - 2 * a *
#            cos(2 * pi * k / N))
#        fft_theor = pk(freq, alpha, n0)
#        fft_theor = variance * fft_theor     # Including time-series variance
#        signif = fft_theor
#
#        try:
#            if dof == -1:
#                dof = dofmin
#        except:
#            pass
#
#        if sigma_test == 0:  # No smoothing, dof=dofmin, TC98 sec. 4
#            dof = dofmin
#            # As in Torrence and Compo (1998), equation 18
#            chisquare = chi2.ppf(significance_level, dof) / dof
#            signif = fft_theor * chisquare
#        elif sigma_test == 1:  # Time-averaged significance
#            if len(dof) == 1:
#                dof = zeros(1, J+1) + dof
#            sel = find(dof < 1)
#            dof[sel] = 1
#            # As in Torrence and Compo (1998), equation 23:
#            dof = dofmin * (1 + (dof * dt / gamma_fac / scales) ** 2 ) ** 0.5
#            sel = find(dof < dofmin)
#            dof[sel] = dofmin  # Minimum dof is dofmin
#            for n, d in enumerate(dof):
#                chisquare = chi2.ppf(significance_level, d) / d;
#                signif[n] = fft_theor[n] * chisquare
#        elif sigma_test == 2:  # Time-averaged significance
#            if len(dof) != 2:
#                raise Exception, ('DOF must be set to [s1, s2], '
#                                  'the range of scale-averages')
#            if Cdelta == -1:
#                raise Exception, ('Cdelta and dj0 not defined for %s with f0=%f' %
#                                 (wavelet.name, wavelet.f0))
#
#            s1, s2 = dof
#            sel = find((scales >= s1) & (scales <= s2));
#            navg = sel.size
#            if navg == 0:
#                raise Exception, 'No valid scales between %d and %d.' % (s1, s2)
#
#            # As in Torrence and Compo (1998), equation 25
#            Savg = 1 / sum(1. / scales[sel])
#            # Power-of-two mid point:
#            Smid = exp((log(s1) + log(s2)) / 2.)
#            # As in Torrence and Compo (1998), equation 28
#            dof = (dofmin * navg * Savg / Smid) * ((1 + (navg * dj / dj0) ** 2) **
#                                                  0.5)
#            # As in Torrence and Compo (1998), equation 27
#            fft_theor = Savg * sum(fft_theor[sel] / scales[sel])
#            chisquare = chi2.ppf(significance_level, dof) / dof;
#            # As in Torrence and Compo (1998), equation 26
#            signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare
#        else:
#            raise Exception, 'sigma_test must be either 0, 1, or 2.'
#
#        return (signif, fft_theor)
#
#    def plotWavelet(self,signal, title, label, units, \
#                                mother = Morlet(6.), t0=1.0,\
#                                dt=1.0, dj=0.25, s0=-1, \
#                                J=-1, alpha=0.0, slevel = 0.95,\
#                                avg1 =15, avg2=20, nameSave=None):
#        """
#        Plot Wavelet Transfor for one signal
#
#PARAMETER:
#    signal : The signal that will be transformed
#    title : Title of the plot
#    label : Label
#    units : unit of the data
#    mother : The Mother  Wavelet. Default Morlet mother wavelet with wavenumber=6
#    t0 : Initial time step
#    dt : time step
#    dj : Four sub-octaves per octaves
#    s0 : Starting scale, here 6 months
#    J : Seven powers of two with dj sub-octaves
#    alpha: Lag-1 autocorrelation for white noise
#    slevel : Significance level
#    avg1,avg2 :  Range of periods to average
#    nameSave : Path plus name to save the plot
#        """
#
#        var = signal
#
#
#        std = var.std()                      # Standard deviation
#        std2 = std ** 2                      # Variance
#        var = (var - var.mean()) / std       # Calculating anomaly and normalizing
#
#        N = var.size                         # Number of measurements
#        time = numpy.arange(0, N) * dt + t0  # Time array in years
#
#        dj = 0.25                            # Four sub-octaves per octaves
#        s0 = -1 #2 * dt                      # Starting scale, here 6 months
#        J = -1 # 7 / dj                      # Seven powers of two with dj sub-octaves
#        alpha = 0.0                          # Lag-1 autocorrelation for white noise
#
#        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0, J,
#                                                              mother)
#        iwave = wavelet.icwt(wave, scales, dt, dj, mother)
#        power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
#        fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
#        period = 1. / freqs
#
#        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
#                                significance_level=slevel, wavelet=mother)
#        sig95 = (signif * numpy.ones((N, 1))).transpose()
#        sig95 = power / sig95                # Where ratio > 1, power is significant
#
#        # Calculates the global wavelet spectrum and determines its significance level.
#        glbl_power = std2 * power.mean(axis=1)
#        dof = N - scales                     # Correction for padding at edges
#        glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
#                               significance_level=slevel, dof=dof, wavelet=mother)
#
#        # Scale average between avg1 and avg2 periods and significance level
#        sel = pylab.find((period >= avg1) & (period < avg2))
#        Cdelta = mother.cdelta
#        scale_avg = (scales * numpy.ones((N, 1))).transpose()
#        # As in Torrence and Compo (1998) equation 24
#        scale_avg = power / scale_avg
#        scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
#        scale_avg_signif, tmp = wavelet.significance(std2, dt, scales, 2, alpha,
#                                    significance_level=slevel, dof=[scales[sel[0]],
#                                    scales[sel[-1]]], wavelet=mother)
#
#        # The following routines plot the results in four different subplots containing
#        # the original series anomaly, the wavelet power spectrum, the global wavelet
#        # and Fourier spectra and finally the range averaged wavelet spectrum. In all
#        # sub-plots the significance levels are either included as dotted lines or as
#        # filled contour lines.
#        pylab.close('all')
#        fontsize = 'medium'
#        params = {'text.fontsize': fontsize,
#                  'xtick.labelsize': fontsize,
#                  'ytick.labelsize': fontsize,
#                  'axes.titlesize': fontsize,
#                  'text.usetex': True
#                 }
#        pylab.rcParams.update(params)          # Plot parameters
#        figprops = dict(figsize=(11, 8), dpi=72)
#        fig = pylab.figure(**figprops)
#
#        # First sub-plot, the original time series anomaly.
#        ax = pylab.axes([0.1, 0.75, 0.65, 0.2])
#        #ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
#        ax.plot(time, var, 'k', linewidth=1.5)
#        ax.set_title('a) %s' % (title, ))
#        if units != '':
#          ax.set_ylabel(r'%s [$%s$/$%s$]' % (label, units,units, ))
#        else:
#          ax.set_ylabel(r'%s' % (label, ))
#
#        # Second sub-plot, the normalized wavelet power spectrum and significance level
#        # contour lines and cone of influece hatched area.
#        bx = pylab.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
#        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
#        bx.contourf(time, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
#                    extend='both')
#        bx.contour(time, numpy.log2(period), sig95, [-99, 1], colors='k',
#                   linewidths=2.)
#        bx.fill(numpy.concatenate([time[:1]-dt, time, time[-1:]+dt, time[-1:]+dt,
#                time[:1]-dt, time[:1]-dt]), numpy.log2(numpy.concatenate([[1e-9], coi,
#                [1e-9], period[-1:], period[-1:], [1e-9]])), 'k', alpha='0.3',
#                hatch='x')
#        bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
#        bx.set_ylabel('Period (Days)')
#        Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
#                                   numpy.ceil(numpy.log2(period.max())))
#        bx.set_yticks(numpy.log2(Yticks))
#        bx.set_yticklabels(Yticks)
#        bx.invert_yaxis()
#
#        # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
#        # noise spectra.
#        cx = pylab.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
#        cx.plot(glbl_signif, numpy.log2(period), 'k--')
#        cx.plot(fft_power, numpy.log2(1./fftfreqs), '-', color=[0.7, 0.7, 0.7],
#                linewidth=1.)
#        cx.plot(glbl_power, numpy.log2(period), 'k-', linewidth=1.5)
#        cx.set_title('c) Global Wavelet Spectrum')
#        if units != '':
#          cx.set_xlabel(r'Power [$%s^2$]' % (units, ))
#        else:
#          cx.set_xlabel(r'Power')
#        #cx.set_xlim([0, glbl_power.max() + std2])
#        cx.set_ylim(numpy.log2([period.min(), period.max()]))
#        cx.set_yticks(numpy.log2(Yticks))
#        cx.set_yticklabels(Yticks)
#        pylab.setp(cx.get_yticklabels(), visible=False)
#        cx.invert_yaxis()
#
#        # Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
#        # avg1 and avg2 parameters
#        dx = pylab.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
#        dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
#        dx.plot(time, scale_avg, 'k-', linewidth=1.5)
#        dx.set_title('d) $%d$-$%d$ year scale-averaged power' % (avg1, avg2))
#        dx.set_xlabel('Time (days)')
#        if units != '':
#          dx.set_ylabel(r'Average variance [$%s$]' % (units, ))
#        else:
#          dx.set_ylabel(r'Average variance')
#        #
#        ax.set_xlim([time.min(), time.max()])
#        #
#        pylab.draw()
#        if(nameSave):
#            pylab.savefig(nameSave)
#        else:
#            pylab.show()
#
#
#
#
#
#
