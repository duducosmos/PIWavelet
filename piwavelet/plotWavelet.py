#!/usr/bin/env python
# *-* Coding: Utf-8 *-*

from __future__ import division

__authors__ = 'Eduardo dos Santos Pereira'
__data__ = '17/01/2015'
__email__ = 'pereira.somoza@gmail.com'

import pylab
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from piwavelet import cwt, significance
from motherWavelets import Morlet

from numpy import arange, ones, where, log2, concatenate, ceil, abs, argsort


class PlotWavelet:
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
 waxQT: qt ax Figure for wavelet plot
 paxQT: qt ax Figure for global power spectrum
    """

    def __init__(self, signal, title, label, units, **kwargs):

        self.signal = signal
        self.title = title
        self.label = label
        self.units = units
        self.fig = None

        self.listParameters = ['mother', 't0', 'dt', 'dj', 's0', 'J', 'alpha',
                      'slevel', 'avg1', 'avg2', 'plotAv', 'fontsize',
                      'labelsize', 'labelpowelog', 'nameSave', 'showFig',
                      'xtickDate', 'waxQT', 'paxQT'
                      ]

        testeKeysArgs = [Ki for Ki in kwargs.keys()
                         if Ki not in self.listParameters]

        if(len(testeKeysArgs) >= 1):
            raise NameError('The keys %s not exist' % testeKeysArgs)

        if 'mother' in kwargs.keys():
            self.mother = kwargs['kwarags']
        else:
            self.mother = Morlet(6.)

        if 't0' in kwargs.keys():
            self.t0 = kwargs['t0']
        else:
            self.t0 = 1.0

        if 'dt' in kwargs.keys():
            self.dt = kwargs['dt']

        else:
            self.dt = 1.0

        if 'dj' in kwargs.keys():
            self.dj = kwargs['dj']
        else:
            self.dj = 0.25

        if 's0'in kwargs.keys():
            self.s0 = kwargs['s0']
        else:
            self.s0 = -1

        if 'J' in kwargs.keys():
            self.J = kwargs['J']
        else:
            self.J = -1

        if 'alpha' in kwargs.keys():
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.0

        if 'slevel' in kwargs.keys():
            self.slevel = kwargs['slevel']
        else:
            self.slevel = 0.95

        if 'avg1' in kwargs.keys():
            self.avg1 = kwargs['avg1']
        else:
            self.avg1 = 15

        if 'avg2' in kwargs.keys():
            self.avg2 = kwargs['avg2']
        else:
            self.avg2 = 20

        if 'plotAv' in kwargs.keys():
            self.plotAv = kwargs['plotAv']
        else:
            self.plotAv = 0

        if 'nameSave' in kwargs.keys():
            self.nameSave = kwargs['nameSave']
        else:
            self.nameSave = None

        if 'fontsize' in kwargs.keys():
            self.fontsize = kwargs['fontsize']
        else:
            self.fontsize = 15

        if 'labelsize' in kwargs.keys():
            self.labelsize = kwargs['labelsize']
        else:
            self.labelsize = 18

        if 'labelpowelog' in kwargs.keys():
            self.labelpowelog = kwargs['labelpowelog']
        else:
            self.labelpowelog = False

        if 'showFig' in kwargs.keys():
            self.showFig = kwargs['showFig']
        else:
            self.showFig = True

        if('xtickDate' in kwargs.keys()):
            self.xtickDate = kwargs['xtickDate']
        else:
            self.xtickDate = None

        if('waxQT' in kwargs.keys()):
            self.axQT = kwargs['waxQT']
        else:
            self.axQT = None

        if('paxQT' in kwargs.keys()):
            self.bxQT = kwargs['paxQT']
        else:
            self.bxQT = None

        self.fft = None
        self.fftfreqs = None

        self._startData()
        self._cwt()

    def __call__(self):
        self.plotWavelet()
        return self.fig

    def _startData(self):
        self.var = self.signal
        # Standard deviation
        self.std = self.var.std()
        # Variance
        self.std2 = self.std ** 2
        # Calculating anomaly and normalizing
        self.var = (self.var - self.var.mean()) / self.std
        # Number of measurementss
        self.N = self.var.size

        if(self.xtickDate is not None):
            self.time = mdates.date2num(self.xtickDate)
        else:
            # Time array in years
            self.time = arange(0, self.N) * self.dt + self.t0

        # Four sub-octaves per octaves
        self.dj = 0.25
        # Starting scale, here 6 months
        self.s0 = -1
        # Seven powers of two with dj sub-octaves
        self.J = -1
        # Lag-1 autocorrelation for white noise
        self.alpha = 0.0

    def _setupPlot(self):
        matplotlib.rcParams['font.size'] = self.fontsize
        matplotlib.rcParams['axes.labelsize'] = self.labelsize
        fontsize = 'medium'
        params = {'text.fontsize': fontsize,
                  'xtick.labelsize': fontsize,
                  'ytick.labelsize': fontsize,
                  'axes.titlesize': fontsize
                  }
        pylab.rcParams.update(params)          # Plot parameters
        figprops = dict(figsize=(11, 8), dpi=72)
        self.fig = pylab.figure(**figprops)

    def _cwt(self):
        wave, scales, freqs, coi, fft, fftfreqs = cwt(self.var,
                                                      self.dt,
                                                      self.dj,
                                                      self.s0,
                                                      self.J,
                                                      self.mother
                                                      )
        #iwave = icwt(wave, self.scales, self.dt, self.dj, self.mother)

        self.fft = fft
        self.fftfreqs = fftfreqs

        power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
        fft_power = self.std2 * abs(fft) ** 2     # FFT power spectrum

        period = 1. / freqs

        signif, fft_theor = significance(1.0, self.dt, scales, 0, self.alpha,
                                        significance_level=self.slevel,
                                        wavelet=self.mother)
        sig95 = (signif * ones((self.N, 1))).transpose()
        # Where ratio > 1, power is significant
        sig95 = power / sig95

        # Calculates the global wavelet spectrum and
        #determines its significance level.
        glbl_power = self.std2 * power.mean(axis=1)
        dof = self.N - scales           # Correction for padding at edges
        glbl_signif, tmp = significance(self.std2,
                                        self.dt,
                                        scales,
                                        1,
                                        self.alpha,
                                        significance_level=self.slevel,
                                        dof=dof,
                                        wavelet=self.mother
                                        )

        # Scale average between avg1 and avg2 periods and significance level
        sel = pylab.find((period >= self.avg1) & (period < self.avg2))
        Cdelta = self.mother.cdelta
        scale_avg = (scales * ones((self.N, 1))).transpose()
        # As in Torrence and Compo (1998) equation 24
        scale_avg = power / scale_avg
        scale_avg = self.std2 * self.dj * self.dt\
                    / Cdelta * scale_avg[sel, :].sum(axis=0)
        scale_avg_signif, tmp = significance(self.std2,
                                             self.dt,
                                             scales,
                                             2,
                                             self.alpha,
                                             significance_level=self.slevel,
                                             dof=[scales[sel[0]],
                                             scales[sel[-1]]],
                                             wavelet=self.mother
                                             )
        self.coi = coi
        self.period = period
        self.power = power
        self.sig95 = sig95
        self.glbl_signif = glbl_signif
        self.fft_power = fft_power
        self.fftfreqs = fftfreqs
        self.scale_avg_signif = scale_avg_signif
        self.scale_avg = scale_avg
        self.glbl_power = glbl_power

    def plotFftSpectrum(self):
        ps = abs(self.fft) ** 2.0
        idx = argsort(self.fftfreqs)
        plt.plot(1.0 / self.fftfreqs[idx], ps[idx])
        plt.title(self.label)
        plt.xlabel("Periodo")
        plt.ylabel("Amplitude")

    def plotWavelet(self):
        """
The following routines plot the results in four different subplots containing
the original series anomaly, the wavelet power spectrum, the global wavelet
and Fourier spectra and finally the range averaged wavelet spectrum. In all
sub-plots the significance levels are either included as dotted lines or as
filled contour lines.
        """

        a = 0
        if(self.axQT is not None):
            self._bxPlot()
            a += 1
        if(self.bxQT is not None):
            self._cxPlot()
            a += 1

        if(a > 0):
            return

        self._setupPlot()
        self._axPlot()
        self._bxPlot()
        self._cxPlot()
        self._dxPlot()
        self._endingPlot()

    def _axPlot(self):
        # First sub-plot, the original time series anomaly.

        self.ax = pylab.axes([0.1, 0.75, 0.65, 0.2])

        self.ax.plot(self.time, self.var, 'k', linewidth=1.5)
        self.ax.set_title('a) %s' % (self.title, ))
        if(self.xtickDate is not None):
            self.ax.xaxis_date()
            self.ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        if self.units != '':
            self.ax.set_ylabel(r'%s [$%s$/$%s$]' % (self.label, self.units,
                                               self.units, ))
        else:
            self.ax.set_ylabel(r'%s' % (self.label, ))

    def _bxPlot(self):

        #Bug...
        #BAD SMELL.
        #Why is zero the last term of the coi
        # when the signal is filtered?
        #... Bug.***:***
        tmpI = where(self.coi == 0)
        if(len(tmpI[0]) != 0):
            if(tmpI[0] == len(self.coi) - 1):
                self.coi[tmpI] = 0.1 * self.coi[-2]

        # Second sub-plot, the normalized wavelet power spectrum
        #and significance level
        # contour lines and cone of influece hatched area.
        if(self.axQT is None):
            self.bx = pylab.axes([0.1, 0.37, 0.65, 0.28], sharex=self.ax)
        else:
            self.bx = self.axQT

        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        self.bx.contourf(self.time, log2(self.period),
                    log2(self.power), log2(levels),
                    extend='both')
        self.bx.contour(self.time, log2(self.period), self.sig95, [-99, 1],
                   colors='k',
                   linewidths=2.)
        self.bx.fill(concatenate([
                self.time[:1] - self.dt,
                self.time,
                self.time[-1:] + self.dt,
                self.time[-1:] + self.dt,
                self.time[:1] - self.dt,
                self.time[:1] - self.dt]),
                log2(concatenate([[1e-9],
                     self.coi,
                     [1e-9],
                     self.period[-1:],
                     self.period[-1:],
                     [1e-9]])),
                'k',
                alpha=0.3,
                hatch='x'
                )
        self.bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (self.label,
                                                            self.mother.name))
        self.bx.set_ylabel('Period (%s)' % self.units)
        self.Yticks = 2 ** arange(ceil(log2(self.period.min())),
                                   ceil(log2(self.period.max())))
        self.bx.set_yticks(log2(self.Yticks))
        self.bx.set_yticklabels(self.Yticks)

        if(self.axQT is not None):
            self.bx.set_ylim([log2(self.Yticks).min(),
                               log2(self.Yticks).max()])
        self.bx.invert_yaxis()
        if(self.xtickDate is not None):
            self.bx.xaxis_date()
            self.bx.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    def _cxPlot(self):

        # Third sub-plot, the global wavelet and Fourier power
        #spectra and theoretical
        # noise spectra.
        visible = False
        if(self.bxQT is not None):
            self.cx = self.bxQT
            visible = True
        else:
            self.cx = pylab.axes([0.77, 0.37, 0.2, 0.28], sharey=self.bx)
        self.cx.plot(self.glbl_signif, log2(self.period), 'k--')
        self.cx.plot(self.fft_power,
                log2(1. / self.fftfreqs),
                '-',
                color=[0.7, 0.7, 0.7],
                linewidth=1.)
        self.cx.plot(self.glbl_power, log2(self.period), 'k-', linewidth=1.5)
        self.cx.set_title('c) Global Wavelet Spectrum')
        if self.units != '':
            self.cx.set_xlabel(r'Power [$%s^2$]' % (self.units, ))
        else:
            self.cx.set_xlabel(r'Power')

        if(self.labelpowelog):
            self.cx.set_xscale('log')
        #cx.set_xlim([0, glbl_power.max() + std2])
        self.cx.set_ylim(log2([self.period.min(), self.period.max()]))
        self.cx.set_yticks(log2(self.Yticks))
        self.cx.set_yticklabels(self.Yticks)
        pylab.setp(self.cx.get_yticklabels(), visible=visible)
        self.cx.invert_yaxis()

    def _dxPlot(self):

        if(self.plotAv == 1):
            # Fourth sub-plot, the scale averaged wavelet spectrum as
            #determined by the
            # avg1 and avg2 parameters
            self.dx = pylab.axes([0.1, 0.07, 0.65, 0.2], sharex=self.ax)
            self.dx.axhline(self.scale_avg_signif,
                       color='k',
                       linestyle='--',
                       linewidth=1.)
            self.dx.plot(self.time,
                    self.scale_avg,
                    'k-',
                    linewidth=1.5
                    )
            self.dx.set_title('d) $%d$-$%d$ year scale-averaged power'
                         % (self.avg1, self.avg2))
            self.dx.set_xlabel('Time (%s)' % self.units)
            if self.units != '':
                self.dx.set_ylabel(r'Average variance [$%s$]' % (self.units, ))
            else:
                self.dx.set_ylabel(r'Average variance')
        else:
            self.bx.set_xlabel('Time (days)')

    def _endingPlot(self):

        self.ax.set_xlim([self.time.min(), self.time.max()])
        pylab.draw()
        if(self.nameSave):
            pylab.savefig(self.nameSave)

        if(self.showFig):
            pylab.show()
