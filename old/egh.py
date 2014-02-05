# -*- coding: utf-8 -*-
'''
Exponential-Gaussian Hybrid (EGH)
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
  expFactor = np.exp((-1.0*(t-tR)**2.0)/(2.0*w*w+tau*(t-tR)))
  y = np.where(2.0*w*w+tau*(t-tR)>0,H*expFactor,0.0)
  return y

cFig=plt.figure()
cAx=cFig.add_subplot(111)
x=np.linspace(1,1000,1000)
y=egh(x,10.0,100.0,20.0,50.0)
cAx.plot(x,y)
plt.show()
plt.close()