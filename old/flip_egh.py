# -*- coding: utf-8 -*-
'''
Flipped Exponential-Gaussian Hybrid (EGH)
Lan et al 2001
'''
import numpy as np
import matplotlib.pyplot as plt

def egh(t,H,tR,w,tau):
  '''
  H = magnitude of peak maximum
  tR = retention time
  w = std. dev of precursor gaussian
  tau = time constant of precursor exponential
  '''
  expFactor = np.exp((-1.0*(tR-t)**2.0)/(2.0*w*w+tau*(tR-t)))
  y = np.where(2.0*w*w+tau*(tR-t)>0,H*expFactor,0.0)
  return y

cFig=plt.figure()
cAx=cFig.add_subplot(111)
x=np.linspace(2600,3800,1000)
y=egh(x,10.0,3100.0,50.0,100.0)
cAx.plot(x,y)
plt.show()
plt.close()