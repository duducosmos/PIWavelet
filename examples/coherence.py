from piwavelet.piwavelet import wcoherence, wcross, plotWavelet
import numpy as np
n=100
y1 = np.random.rand(n)
y2 = np.random.rand(n)
x = np.arange(0, n, 1)
y1 = (y1-y1.mean())/y1.std()
y2 = (y2-y2.mean())/y2.std()
mc = wcoherence(y1, y2)
mc.plot(t=x, title="Test", units="sec")
myXSpec = wcross(x,y1)
myXSpec.plot(t = x, title='My Title',units='year')
plotWavelet(y1, title='test',label='Random Signal',units='days')
input("Enter para seguir")
