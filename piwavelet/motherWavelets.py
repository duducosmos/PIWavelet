#!/usr/bin/env python
# *-* Coding: Utf-8 *-*

from __future__ import division

__authors__ = 'Eduardo dos Santos Pereira'
__data__ = '17/01/2015'
__email__ = 'pereira.somoza@gmail.com'

from numpy import pi, exp, sqrt, prod
from scipy.special import gamma
from scipy.special.orthogonal import hermitenorm
from numpy.lib.polynomial import polyval


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
        return 1. / self.coi()

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
        return 1. / self.coi()

    def _set_m(self, m):
        # Sets the m derivative of a Gaussian, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.m = m               # Wavelet order
        self.dofmin = 2         # Minimum degrees of freedom
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
        return 1. / self.coi()

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
