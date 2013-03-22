#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from piwavelet import piwavelet
y1 = np.random.rand(50) #Generation of the Random Signal
x = np.arange(0,50,1) # Time step
plt.plot(x,y1,label='y')
plt.legend(loc=4)
plt.show()
plt.clf()
y1 = (y1-y1.mean())/y1.std() #Normalization of the Signal 1
plt.plot(x,y1,label='y1')
plt.legend(loc=4)
plt.show()
wave, scales, freqs, coi, fft, fftfreqs=piwavelet.cwt(y1) # If you want to know the individual properties.'
piwavelet.plotWavelet(y1,title='test',label='Random Signal',units='days')
raw_input('Enter to finish')

