#!/usr/bin/env python
# *-* Coding: Utf-8 *-*

from __future__ import division

__authors__ = 'Eduardo dos Santos Pereira'
__data__ = '17/01/2015'
__email__ = 'pereira.somoza@gmail.com'

from oct2py import octave
import os
from numpy import pi, angle, cos, sin, log2, ceil, arange, concatenate

import pylab
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime


class Wcoherence:
    """
    This class is an Python interface for the Wavelet Coherence matlab
    functions of the package for wavelet,
    cross-wavelet and coherence-wavelet analises profided by
    Aslak Grinsted, John C. Moore and Svetlana Jevrejeva.

    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence

    However, the Continuous wavelet transform of the signal, in this class,
    is a pure python
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
        mFiles = HOME + '/.piwavelet/wtc/'
        self.wtcPath = octave.addpath(mFiles)
        self.signal1 = signal1
        self.signal2 = signal2
        self.Rqs, self.period, self.scale, self.coi,\
        self.wtcsig = self.__wtc(self.signal1, self.signal2)
        self.freqs = 1.0 / self.period

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
        Rqs, period, scale, coi, wtcsig = octave.wtc(signal1, signal2, nout=5)
        period = period[0]
        scale = scale[0]
        coi = coi[0]
        return Rqs, period, scale, coi, wtcsig

    def plot(self, t, title, units, **kwargs):
        """
        Plots the wavelet coherence

            PARAMETERS
                title: Title of the Plot
                units: (string) Units of the period and time  (e.g. 'days')
                t : array with time
                OPTIONALS:
                    gray : (boolean) True for gray map .
                    levels : List with significance level that
                             will be showed in the plot
                    labels : List with the Label of significance level
                             that will be apper into the color bar.
                             If not defined, the levels list is used instead
                    pArrow : (boolean)  True for draw vector of phase angle
                              (it has problem not recomended for
                               large sample of data)
                    pSigma : (boolean) True for draw the significance
                              countor lines
                    nameSave : (string) path plus name to save the figure,
                               if it is define, the plot is saved but not showed
                    scale : (boolean) True  for not log2 scale of the Plot
        """

        return self.__plotWC(wc=self.Rqs, t=t, coi=self.coi,
                             freqs=self.freqs,
                             signif=self.wtcsig,
                             title=title,
                             units=units, **kwargs)

    def __plotWC(self, wc, t, coi,
                 freqs, signif, title, units='days', **kwargs):
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
                    levels : List with significance level that
                             will be showed in the plot
                    labels : List with the Label of significance level
                             that will be apper into the color bar.
                             If not defined, the levels list is used instead
                    pArrow : (boolean)  True for draw vector of phase
                             angle (it has problem not
                             recomended for large sample of data)
                    pSigma : (boolean) True for draw the
                              significance countor lines
                    nameSave : (string) path plus name to
                                save the figure, if it is define, the
                                plot is saved but not showed
                    scale : (boolean) True  for not log2 scale of the Plot


        RETURNS
            A list with the figure and axis objects for the plot.



        """

        listParameters = ['levels', 'labels', 'pArrow', 'pSigma', 'gray',
                          'nameSave', 'scale', 'zoom', 'labelsize', 'fontsize']

        testeKeysArgs = [Ki for Ki in kwargs.keys()
                         if Ki not in  listParameters]

        if(len(testeKeysArgs) >= 1):
            raise NameError('The key %s are not defined: ' % testeKeysArgs)

        # Sets some parameters and renames some of the input variables.
        from matplotlib import pyplot

        if 'levels' in kwargs.keys():
            levels = kwargs['levels']
        else:
            levels = None

        if 'labels' in kwargs.keys():
            labels = kwargs['labels']
        else:
            labels = None

        if 'pArrow' in kwargs.keys():
            pArrow = kwargs['pArrow']
        else:
            pArrow = None

        if 'pSigma' in kwargs.keys():
            pSigma = kwargs['pSigma']
        else:
            pSigma = True

        if 'gray' in kwargs.keys():
            gray = kwargs['gray']
        else:
            gray = None

        if 'nameSave' in kwargs.keys():
            nameSave = kwargs['nameSave']
        else:
            nameSave = None

        if 'scale' in kwargs.keys():
            scale = kwargs['scale']
        else:
            scale = 'log2'

        if 'zoom' in kwargs.keys():
            if(len(kwargs['zoom']) <= 1 or len(kwargs['zoom']) > 2):
                zoom = None
            else:
                zoom = kwargs['zoom']
        else:
            zoom = None

        if 'fontsize' in kwargs.keys():
            fontsize = kwargs['fontsize']
        else:
            fontsize = 18

        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (10, 10 / 1.61803398875)

        if('labelsize' in kwargs.keys()):
            labelsize = kwargs['labelsize']
            labelsize = int(labelsize)

        else:
            labelsize = 15

        params = {'font.family': 'serif',
                          'font.sans-serif': ['Helvetica'],
                          'font.size': fontsize,
                          'figure.figsize': figsize,
                          'font.stretch': 'ultra-condensed',
                          'xtick.labelsize': labelsize,
                          'ytick.labelsize': labelsize,
                          'axes.titlesize': fontsize,
                          'timezone': 'UTC'
                         }
        pyplot.rcParams.update(params)

        if(nameSave is None):
            pyplot.ion()
        else:
            pyplot.ioff()
        fp = dict()
        ap = dict(left=0.15, bottom=0.12, right=0.95, top=0.95,
                     wspace=0.10, hspace=0.10)
        #orientation='landscape'
        fig = pyplot.figure(**fp)
        fig.subplots_adjust(**ap)

        timeDT = False
        try:
            from pandas.tslib import Timestamp
            if(type(t[0]) == Timestamp):
                timeDT = True
                t = mdates.date2num(t)
        except:
            pass

        if(type(t[0]) == datetime.datetime):
            timeDT = True
            t = mdates.date2num(t)

        #N = len(t)
        dt = t[1] - t[0]
        period = 1. / freqs
        power = wc

        # power is significant where ratio > 1
        sig95 = signif

        # Calculates the phase between both time series. The phase arrows in the
        # cross wavelet power spectrum rotate clockwise with 'north' origin.
        agl = 0.5 * pi - angle(wc)
        u, v = cos(agl), sin(agl)

        result = []

        da = [3, 3]

        fig = fig
        result.append(fig)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('%s' % title)
        ax.set_xlabel('Time (%s)' % units)
        ax.set_ylabel('Period (%s)' % units)
        if(timeDT):
            ax.xaxis_date()
            ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
            fig.autofmt_xdate(bottom=0.18)

        # Plots the cross wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.

        if(levels):
            if(labels):
                pass
            else:
                labels = [str(li) for li in levels]
        else:
            levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6',
                       '0.7', '0.8', '0.9', '1']
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
            Power = log2(power)
            Levels = log2(levels)
        else:
            Power = power
            Levels = levels

        norml = colors.BoundaryNorm(Levels, 256)

        if(gray is True):
            cf = ax.contourf(t, log2(period), Power, Levels,
                cmap=plt.cm.gray, norm=norml, extend=extend)
        else:
            cf = ax.contourf(t, log2(period), Power, Levels,
            cmap=plt.cm.jet, norm=norml, extend=extend)

        if(pSigma):
            ax.contour(t, log2(period), sig95, [-99, 1], colors='k',
                linewidths=2.)

        if(pArrow):
            ax.quiver(t[::da[1]], log2(period)[::da[0]],
                 u[::da[0], ::da[1]],
                v[::da[0], ::da[1]], units='width', angles='uv', pivot='mid'
                #linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
                #headaxislength=5, minshaft=2, minlength=5
                )

        if(zoom):
            newPeriod = period[pylab.find((period >= zoom[0]) &
                               (period <= zoom[1]))]
            ax.fill(concatenate([t[:1] - dt, t, t[-1:] + dt, t[-1:] + dt,
                                 t[:1] - dt, t[:1] - dt]),
                     log2(concatenate([[1e-9], coi, [1e-9],
                                        period[-1:], period[-1:], [1e-9]])),
                     'k', alpha=0.3, hatch='x')
            Yticks = 2 ** arange(ceil(log2(period.min())),
                ceil(log2(period.max())))
            ax.set_yticks(log2(Yticks))
            ax.set_yticklabels(Yticks)
            ax.set_xlim([t.min(), t.max()])
            ax.set_ylim(log2([newPeriod.min(),
                              min([coi.max(), newPeriod.max()])
                              ]))
            ax.invert_yaxis()
            cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
            cbar.ax.set_yticklabels(labels)

            pylab.draw()

        else:

            ax.fill(concatenate([t[:1] - dt, t, t[-1:] + dt, t[-1:] + dt,
                                 t[:1] - dt, t[:1] - dt]),
                    log2(concatenate([[1e-9], coi, [1e-9],
                                      period[-1:], period[-1:], [1e-9]])),
                    'k',
                    alpha=0.3,
                    hatch='x'
                    )
            Yticks = 2 ** arange(ceil(log2(period.min())),
                ceil(log2(period.max())))
            ax.set_yticks(log2(Yticks))
            ax.set_yticklabels(Yticks)
            ax.set_xlim([t.min(), t.max()])
            ax.set_ylim(log2([period.min(), min([coi.max(), period.max()])]))
            ax.invert_yaxis()
            cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
            cbar.ax.set_yticklabels(labels)

            pylab.draw()

        if nameSave is not None:
            pylab.savefig(nameSave, dpi=80)
        else:
            plt.show()

        result.append(ax)

        return result
