#!/usr/bin/env python
# -*- Coding: UTF-8 -*-
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
import sys,os
import piwavelet
import shutil


HOME = os.path.expanduser('~')

if not os.path.exists(HOME+'/.piwavelet'):

    local = os.path.dirname(piwavelet.__file__)
    print 'Creating .piwavelet cache diretory in %s' %HOME
    os.makedirs(HOME+'/.piwavelet')
    shutil.copytree(local+'/wtc',HOME+'/.piwavelet/wtc')

__all__ = ['waveletCC', 'Morlet', 'Paul', 'DOG',
           'Mexican_hat']
