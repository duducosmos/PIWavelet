#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from piwavelet import piwavelet
y1 = np.random.rand(50) #Generation of the Random Signal 1
y2 = np.random.rand(50) #Generation of the Random Signal 2
x = np.arange(0,50,1) # Time step
plt.plot(x,y1,label='y1')
plt.plot(x,y2,label='y2')
plt.legend(loc=4)
plt.show()
plt.clf()
y1 = (y1-y1.mean())/y1.std() #Normalization of the Signal 1
y2 = (y2-y2.mean())/y2.std() #Normalization of the Signal 2
plt.plot(x,y1,label='y1')
plt.plot(x,y2,label='y2')
plt.legend(loc=4)
plt.show()
myXSpec = piwavelet.wcross(y1,y2) #Cross Wavelet Analisys
myXSpec.plot(t = x, title='Test',units='sec') # Plot of the Cross Power Spectrum Map
xwt,period, scale, coi, sig95=myXSpec() # If you want to know the individual properties.'
raw_input('Enter to finish')

