import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import convolution
import warnings
import pickle
import os, sys
import utils
from copy import deepcopy

class baseSpectrum:
  """ Generic 1d spectrum class. """
  def __init__(self,x,y,specname='Unknown spectrum',nonData=[]):
    """
    Init requires input of x axis values (wavelength etc.) and y axis values (intensity etc.)
    nonData adds extra variable names to the list of variables to be ignored by
    the fixbad function.
    """
    if len(x) != len(y):                                  #Check that input is sane
      raise RuntimeError('Input arrays have different sizes.')
    if type(x) != u.quantity.Quantity:
      raise u.UnitsError('the x data is not in astropy unit format')
    else:
      self.x=x
    if type(y) != u.quantity.Quantity:
      raise u.UnitsError('the y data is not in astropy unit format')
    else:
      self.y=y
    self.dy=np.nan                                          #Do a baseline to get this
    self.name=str(specname)                                 #Spectrum name
    self.baselined=False                                    #Has the spectrum been baselined?
    self.convolved=False                                    #Has the spectrum been convolved?
    self.drawstyle='steps-mid'
    self.nonData = [
                      'nonData',\
                      'name',\
                      'convolved','baselined',\
                      'drawstyle',\
                      'dy'\
                   ]
    for cNonData in nonData:                                #Add the extra non-array variable names into nonData
      if not cNonData in self.nonData:
        self.nonData.append(cNonData)                       
    self.fixbad()                                          #Drop bad data.
    self.sort()
  def sort(self):
    """
    Sort the data arrays to go in increasing
    order of x
    """
    sorter=np.argsort(self.x)
    nonDatavars = self.nonData
    ownvarnames = self.__dict__.keys()
    ownvarnames = filter (lambda a: not a in nonDatavars, ownvarnames)
    varlength = len(self.__dict__[ownvarnames[0]])
    iGoodones = np.isfinite(np.ones(varlength))
    for cVarname in ownvarnames:
      self.__dict__[cVarname]=self.__dict__[cVarname][sorter]
  def fixbad(self):
    """
    Make spectrum go through its own vars and replace all the bad ones with nans.
    All variable names in nonData are ignored by this function.
    """
    ignorevars = self.nonData
    ownvarnames = self.__dict__.keys()
    ownvarnames = filter (lambda a: a not in ignorevars, ownvarnames)
    varlength = len(self.__dict__[ownvarnames[0]])
    iGoodones = np.isfinite(np.ones(varlength))
    for cVarname in ownvarnames:
      cVar = self.__dict__[cVarname]
      if len(cVar) != varlength:
        raise RuntimeError('Anomalous variable length detected in spectrum!')
      iGoodones = np.logical_and(iGoodones,np.isfinite(cVar))
    iBadones = np.logical_not(iGoodones)
    for cVarname in ownvarnames:
      if cVarname != 'x':
        self.__dict__[cVarname][iBadones]=np.nan
  def plot(self,axis,plotstyle='k-',iDrawstyle=None,x=None,y=None):
    """
    Plot x,y into given MPL axis.
    They can be overridden with different
    variable names.
    """
    if x:
      plotx = self.__dict__[x]
    else:
      plotx = self.x.value
    if y:
      ploty = self.__dict__[y]
    else:
      ploty = self.y.value
    if iDrawstyle:
      drawstyle=iDrawstyle
    else:
      drawstyle=self.drawstyle
    iAx.plot(cX,cY,plotstyle,drawstyle=drawstyle)
  def convert2wn(self):
    """
    Convert x axis to wavenumber [cm^-1]
    """
    with u.set_enabled_equivalencies(u.equivalencies.spectral()):
      self.x=self.x.to(u.kayser)
    self.sort()
  def convert2wl(self):
    """
    Convert x axis to wavelength [um]
    """
    with u.set_enabled_equivalencies(u.equivalencies.spectral()):
      self.x=self.x.to(u.micron)
    self.sort()
  def subspectrum(self,minX,maxX):
    """
    Return a slice of the spectrum as a new spectrum,
    using x axis units as the limits.
    Slice limits are inclusive.
    """
    iSub = np.logical_and(np.greater_equal(self.x,minX),np.less_equal(self.x,maxX))
    newX = self.x[iSub]
    newY = self.y[iSub]
    newSpec = deepcopy(self)
    newSpec.x = newX
    newSpec.y = newY
    return newSpec
  def interpolate(self,targSpectrum):
    """
    Interpolate spectrum to match target spectrum resolution.
    Does not modify current spectrum. Returns a new one.
    New spectrum metadata is taken from target spectrum.
    """
    if self.x.unit != targSpectrum.x.unit:
      raise u.UnitsError('Spectrums have different units on x axis!')
    if self.y.unit != targSpectrum.y.unit:
      raise u.UnitsError('Spectrums have different units on y axis!')
    newX=targSpectrum.x
    newY=np.interp(newX,self.x,self.y)
    newSpec = deepcopy(self)
    newSpec.x = newX
    newSpec.y = newY
    newSpec.name = self.name+'(interpolated: '+targSpectrum.name+')'
    newSpec.baselined = targSpectrum.baselined
    newSpec.convolved = targSpectrum.convolved
    newSpec.drawstyle = targSpectrum.drawstyle
    return newSpec
  def yat(self,x):
    """ Return value of y at x """
    return np.interp(x,self.x,self.y)
  def convolve(self,kernel):
    """ Convolve spectrum y with given kernel """
    if self.convolved:
      warnings.warn('Spectrum '+self.name+' has already been convolved once!',RuntimeWarning)
    self.y=convolution.convolve(self.y,kernel)
    self.convolved=True
  def gconvolve(self,fwhm):
    """ Convolve spectrum using a gaussian of given fwhm (in units of x axis) """
    gkernel=convolution.Gaussian1DKernel(fwhm)
    self.convolve(gkernel)
  def smooth(self,window_len=11,window='hanning'):
    """
    A smoothing function
    adapted from http://stackoverflow.com/questions/5515720/python-smooth-time-series-data
    """
    if self.x.ndim != 1:
      raise ValueError, "smooth only accepts 1 dimension arrays."
    if self.x.size < window_len:
      raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
      self.y = self.x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
      raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*self.x[0]-self.x[window_len-1::-1],self.x,2*self.x[-1]-self.x[-1:-window_len:-1]]
    if window == 'flat': #moving average
      w=np.ones(window_len,'d')
    else:  
      w=eval('np.'+window+'(window_len)')
    self.y=np.convolve(w/w.sum(),s,mode='same')
  def baseline(self,degree=1,windows=[[0.0,1.0e6]],exclusive=False,useFile=None,overWrite=False):
    """
    Correct the y with a new baseline.
    Default mode is inclusive.
    """
    iBaseline=np.logical_or(np.isinf(self.x),exclusive)
    if useFile != None and os.path.exists(useFile):
      with open(useFile,'r') as cFile:
        windows = pickle.load(cFile) 
    elif windows=='manual':
      print 'Determining manual baseline'
      cFig=plt.figure()
      cAx = cFig.add_subplot(111)
      cManager = plt.get_current_fig_manager()
      cManager.window.wm_geometry("+100+50")
      cAx.plot(self.x,self.y,'k-',drawstyle=self.drawstyle)
      cBaseliner = baseliner(cAx,self)
      if not hasattr(sys,'_called_from_test'): #only show the plot if not testing
        plt.show(cFig)
        windows=cBaseliner.windows
      else:
        return cFig,cBaseliner #send the relevant stuff back for testing
      if useFile != None:
        with open(useFile,'w') as cFile:
          pickle.dump(windows,cFile)
        print 'Wrote window data to '+useFile
    for cWindow in windows:
      if exclusive:
        iBaseline=np.logical_and(iBaseline,np.logical_or(np.less(self.x,cWindow[0]),np.greater(self.x,cWindow[1])))
      else:
        iBaseline=np.logical_or(iBaseline,np.logical_and(np.greater(self.x,cWindow[0]),np.less(self.x,cWindow[1])))
    baseline = np.polyfit(self.x[iBaseline],self.y[iBaseline],degree)
    if not(np.all(np.isfinite(baseline))):
      raise RuntimeError('Baseline is non-finite!')
    fixedY = self.y
    for cPower in range(degree+1):
      fixedY=fixedY-baseline[degree-cPower]*self.x**cPower
    self.y=fixedY
    self.dy=np.abs(np.std(fixedY[iBaseline]))
    self.baselined=True
  def shift(self,amount):
    """
    Shifts the spectrum by amount
    specified, in primary x axis
    units.
    """
    self.x+=amount
  def max(self,checkrange=None):
    """
    Returns maximum y of the spectrum.
    If checkrange is set, returns maximum inside of that range.
    """
    iCheckrange=np.ones_like(self.y,dtype=bool)
    if np.any(checkrange):
      minX=checkrange[0]
      maxX=checkrange[1]     
      iCheckrange=np.logical_and(iCheckrange,np.logical_and(
                        np.less_equal(minX,self.x),np.greater_equal(maxX,self.x)))
    return np.nanmax(self.y[iCheckrange])
  def min(self,checkrange=None):
    """
    Returns minimum y of the spectrum.
    If checkrange is set, returns maximum inside of that range.
    """
    iCheckrange=np.ones_like(self.y,dtype=bool)
    if np.any(checkrange):
      minX=checkrange[0]
      maxX=checkrange[1]     
      iCheckrange=np.logical_and(iCheckrange,np.logical_and(
                        np.less_equal(minX,self.x),np.greater_equal(maxX,self.x)))
    return np.nanmin(self.y[iCheckrange])
  def info(self):
    print '---'
    print 'Summary for spectrum '+self.name
    print 'x unit: '+self.xUnit
    print 'min(x): '+str(np.nanmin(self.x))
    print 'max(x): '+str(np.nanmax(self.x))
    print 'y unit: '+self.yUnit
    print 'min(y): '+str(np.nanmin(self.y))
    print 'max(y): '+str(np.nanmax(self.y))
    print 'baselined: '+str(self.baselined)
    print 'convolved: '+str(self.convolved)
    print '---'

class absorptionSpectrum(baseSpectrum):
  """
  An absorption spectrum, with all the
  specific details that involves.
  Units on the y axis are in optical depth
  """
  def __init__(self,iWn,iOd,specname='Unknown absorption spectrum',nonData=[]):
    """
    Init the spectrum. Places iOd on the y axis and
    iWn on the x axis.
    """
    if len(iWn) != len(iOd):
      raise RuntimeError('Input arrays have different sizes.')
    self.od = iOd #Optical depth
    self.wn = iWn #Wave number
    with u.set_enabled_equivalencies(u.equivalencies.spectral()):
      self.wl=self.wn.to(u.micron)
    baseSpectrum.__init__(self,self.wn,self.od,specname=specname,nonData=nonData)
  def plotod(self,iAx,in_wl=False,**kwargs):
    """
    Plot the optical depth spectrum as function of wavenumber
    to given axis unless flag is set.
    """
    if in_wl:
      self.plot(iAx,x='wl',y='od',**kwargs)      
    else:
      self.plot(iAx,x='wn',y='od',**kwargs)


class labSpectrum(absorptionSpectrum):
  """
  Laboratory spectrum class from optical constants.
  Inherits absorptionSpectrum class propertries.
  Does CDE correction to the data.
  """
  def __init__(self,iWn,iN,iK,specname='Unknown CDE-corrected laboratory spectrum'):
    """
    Init requires input of wavenumber array [cm^-1] and od array.
    Optional input: Name of spectrum
    """
    if len(iWn) != len(iN) or len(iK) != len(iN):
      raise RuntimeError('Input arrays have different sizes.')
    self.cabs,self.cabs_vol,self.cscat_vol,self.ctot=utils.cde_correct(iWn,iN,iK)
    self.n=np.array(iN,dtype='float64')
    self.k=np.array(iK,dtype='float64')
    absorptionSpectrum.__init__(self,iWn*u.kayser,self.cabs_vol*utils.unit_od,specname=specname)
  def plotnk(self,xrange=None):
    """ Plot the optical constants as function of wavenumber. Return figure where plotting happened. """
    if np.any(xrange):
      minx=xrange[0]
      maxx=xrange[1]
    else:
      minx=np.min(self.wn)
      maxx=np.max(self.wn)
    c_fig=plt.figure()
    c_axis = c_fig.add_subplot(211)
    c_axis.set_title('n')
    c_axis.set_xlim(minx,maxx)
    c_axis.plot(self.wn,self.n,'k-',drawstyle=self.drawstyle)
    c_axis.plot(self.wn,self.n,'k.',drawstyle=self.drawstyle)
    c_axis = c_fig.add_subplot(212)
    c_axis.set_title('k')
    c_axis.set_xlim(minx,maxx)
    c_axis.plot(self.wn,self.k,'k-',drawstyle=self.drawstyle)
    c_axis.plot(self.wn,self.k,'k.',drawstyle=self.drawstyle)
    return c_fig