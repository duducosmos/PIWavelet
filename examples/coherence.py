from piwavelet.piwavelet import wcoherence
import numpy as np
n=50
y1 = np.random.rand(n)
y2 = np.random.rand(n)
x = np.arange(0, n, 1)
y1 = (y1-y1.mean())/y1.std()
y2 = (y2-y2.mean())/y2.std()
mc = wcoherence(y1, y2)
mc.plot(t=x, title="Test", units="sec")
input("Enter para seguir")
