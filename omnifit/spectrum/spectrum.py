import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import convolution
import warnings
import pickle
import os, sys
from .. import utils
from copy import deepcopy

class BaseSpectrum:
  """
  A class to represent spectroscopic data.
  
  This class is designed to work for spectroscopic data of ices, but 
  may work for other types of spectroscopic data as well.
  This is the most basic version of the class, concerned solely with
  the contents of the x and y attributes.

  Attributes
  ----------
    x : astropy.units.Quantity
      Represents the data on the "x-axis" of the spectrum,
      i.e. usually the wavelength or frequency
    y : astropy.units.Quantity
      Represents the data on the "x-axis" of the spectrum,
      i.e. the flux or optical depth
    dy : NoneType or float
      The uncertainty of y. Can be given during initialisation,
      or automatically calculated during baselining. (default=None)
    specname : string
      The name of the spectrum (default='Unknown spectrum')
    baselined : bool
      Indicates whether the spectrum has been baselined or not
    convolved : bool
      Indicates whether the spectrum has been put through convolution
  """
  def __init__(self,x,y,dy=None,specname='Unknown spectrum',nondata=[]):
    """
    BaseSpectrum(x,y,dy=None,specname='Unknown spectrum',nondata=[])

    Constructor for the BaseSpectrum class. Requires x and y;
    everything else is optional.

    Parameters
    ----------
    x : astropy.units.Quantity or numpy.ndarray
      Represents the data on the "x-axis" of the spectrum.
      This is stored as an astropy quantity and thus it is
      recommended that the class constructor is called with
      such an input. However, the constructor also accepts
      a numpy ndarray, in which case it will try to guess
      the units and then convert the input into an appropriate
      astropy quantity.
      The autodetection assumes the units are in kayser units
      (i.e. reciprocal wavenumbers with the unit cm^-1) if the
      mean of the input array is greater than 1000. Otherwise
      the autodetection assumes the units are in microns.
    y : astropy.units.Quantity or numpy.ndarray
      Represents the data on the "x-axis" of the spectrum.
      This is stored as an astropy quantity and thus it is
      recommended that the class constructor is called with
      such an input. However, the constructor also accepts
      a numpy ndarray, in which case it will assume that
      the units are in optical depth and then convert the 
      input into this astropy quantity.
    dy : float, optional
      The uncertainty of y. If given, this is assumed to be
      the uncertainty of the y axis data in the same units
      as given (or assumed) with the y input. Otherwise
      the uncertainty is left as None during initialisation
      and will be calculated as part of baselining.
    specname : string, optional
      An optional human-readable name can be given to the
      spectrum via this input.
    nondata : list, optional
      If information unrelated to the x and y input data is
      stored in the class instance, the variable names in which
      this information is stored can be given here. This causes
      various internal functions (related to automatic sorting 
      and error-checking) of the class to ignore these
      variables.
      It is not usually necessary for the user to use this input
      during initialisation; it is most often used by children of
      the BaseSpectrum class.

    """
    if len(x) != len(y):                                  #Check that input is sane
      raise RuntimeError('Input arrays have different sizes.')
    if type(x) != u.quantity.Quantity:
      #try to guess x units (between micron and kayser; the most common units) if none given
      if np.mean(x) > 1000.:
        warnings.warn('The x data is not in astropy unit format. Autodetection assumes kayser.',RuntimeWarning)
        self.x=x*u.kayser
      else:
        warnings.warn('The x data is not in astropy unit format. Autodetection assumes micron.',RuntimeWarning)
        self.x=x*u.micron
    else:
      self.x=x
    if type(y) != u.quantity.Quantity:
      warnings.warn('The y data is not in astropy unit format. Assuming optical depth.',RuntimeWarning)
      self.y=y*utils.unit_od
    else:
      self.y=y
    if dy is not None:
      self.dy=dy
    else:
      self.dy=None
    self.name=str(specname)                                 #Spectrum name
    self.baselined=False                                    #Has the spectrum been baselined?
    self.convolved=False                                    #Has the spectrum been convolved?
    self.__nondata = [
                      '_BaseSpectrum__nondata',\
                      'name',\
                      'convolved','baselined',\
                      'dy'\
                   ]
    for cnondata in nondata:                                #Add the extra non-array variable names into nondata
      if not cnondata in self.__nondata:
        self.__nondata.append(cnondata)                       
    self.__fixbad()                                          #Drop bad data.
    self.__sort()
  def __sort(self):
    """
    __sort()

    An internal method which sorts the data arrays so that they
    all go in increasing order of x.

    Parameters
    ----------
    None
    """
    sorter=np.argsort(self.x)
    nondatavars = self.__nondata
    ownvarnames = self.__dict__.keys()
    ownvarnames = filter (lambda a: not a in nondatavars, ownvarnames)
    varlength = len(self.__dict__[ownvarnames[0]])
    iGoodones = np.isfinite(np.ones(varlength))
    for cVarname in ownvarnames:
      self.__dict__[cVarname]=self.__dict__[cVarname][sorter]
  def __fixbad(self):
    """
    __fixbad()

    An internal method which replaces all non-number data (e.g. 
    infinities) in the data arrays with numpy.nan.

    Parameters
    ----------
    None
    """
    ignorevars = self.__nondata
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
  def plot(self,axis,x='x',y='y',*args,**kwargs):
    """
    plot(axis,x='x',y='y',*args,**kwargs)

    Plot the contents of the spectrum into a given matplotlib axis.
    Defaults to the data contained in the x and y attributes, but
    can also plot other data content if instructed to do so.

    Parameters
    ----------
    axis : matplotlib.axis
      The axis which the plot will be generated in.
    x : string, optional
      The name of the variable to be plotted on the x axis.
    y : string, optional
      The name of the variable to be plotted on the x axis.
    *args and **kwargs can be used to pass additional plotting
    parameters to the matplotlib plotting routine, as documented
    in the matplotlib documentation.
    """
    try: #assume it's with astropy units
      plotx = self.__dict__[x].value
    except ValueError:
      plotx = self.__dict__[x]
    try: #assume it's with astropy units
      ploty = self.__dict__[y].value
    except ValueError:
      ploty = self.__dict__[y]
    axis.plot(plotx,ploty,*args,**kwargs)
  def convert2wn(self):
    """
    convert2wn()

    Convert the x axis data to kayser (reciprocal wavenumber) units.
    Re-sort the data afterwards.

    Parameters
    ----------
    None
    """
    self.convert2(u.kayser)
  def convert2wl(self):
    """
    convert2wl()

    Convert the x axis data to wavelength (in microns) units.
    Re-sort the data afterwards.

    Parameters
    ----------
    None
    """
    self.convert2(u.micron)
  def convert2(self,newunit):
    """
    convert2(newunit)

    Convert the x axis data to given spectral units.
    Re-sort the data afterwards.

    Parameters
    ----------
    newunit : astropy.units.core.Unit
      Desired (spectral) unit the x axis data should be
      converted to.
    """
    with u.set_enabled_equivalencies(u.equivalencies.spectral()):
      self.x=self.x.to(newunit)
    self.__sort()
  def subspectrum(self,limit_lower,limit_upper):
    """
    subspectrum(limit_lower,limit_upper)

    Create a copy of the spectrum which is cropped along the x
    axis using the given inclusive limits.

    Parameters
    ----------
    limit_lower : float
      The desired minimum x axis of the cropped spectrum, in
      current units of the spectrum. This limit is inclusive.
    limit_upper : float
      The desired maximum x axis of the cropped spectrum, in
      current units of the spectrum. This limit is inclusive.

    Returns
    -------
    Copy of the spectrum, cropped using the given specifications.
    """
    iSub = np.logical_and(np.greater_equal(self.x.value,limit_lower),np.less_equal(self.x.value,limit_upper))
    newX = self.x[iSub]
    newY = self.y[iSub]
    newSpec = deepcopy(self)
    newSpec.x = newX
    newSpec.y = newY
    return newSpec
  def interpolate(self,target_spectrum):
    """
    interpolate target_spectrum

    Interpolate spectrum to match target spectrum resolution.
    Does not modify current spectrum, but returns a new one, which is
    a copy of the current spectrum but with the interpolated data on
    the x and y fields.
    The target spectrum has to be using the same units on the x and
    y axes as the current spectrum, or the interpolation will fail.

    Parameters
    ----------
    target_spectrum : BaseSpectrum
      The target spectrum which the x axis resolution of the current
      spectrum should be made to match.

    Returns
    -------
    Copy of the spectrum, interpolated to match the target spectrum
    x axis resolution.

    """
    if self.x.unit != target_spectrum.x.unit:
      raise u.UnitsError('Spectrums have different units on x axis!')
    if self.y.unit != target_spectrum.y.unit:
      raise u.UnitsError('Spectrums have different units on y axis!')
    newX=target_spectrum.x
    newY=np.interp(newX,self.x,self.y)
    newSpec = deepcopy(self)
    newSpec.x = newX
    newSpec.y = newY
    newSpec.name = self.name+'(interpolated: '+target_spectrum.name+')'
    return newSpec
  def yat(self,x):
    """
    yat(x)

    Return interpolated value of y at requested x.

    Parameters
    ----------
    x : float
      The x axis coordinate of interest.

    Returns
    -------
    The interpolated value of y at the requested x coordinate.

    """
    return np.interp(x,self.x.value,self.y.value)
  def convolve(self,kernel,*args,**kwargs):
    """
    convolve(kernel,*args,**kwargs)

    Use astropy.convolution.convolve to convolve the y axis data of the
    spectrum with the given kernel.
    Modifies the existing spectrum.

    Parameters
    ----------
    kernel : numpy.ndarray or astropy.convolution.Kernel
      A convolution kernel to feed into the convolution function.
    *args and **kwargs can be used to pass additional plotting
    parameters to the astropy convolution routine, as documented
    in the astropy documentation.
    """
    if self.convolved:
      warnings.warn('Spectrum '+self.name+' has already been convolved once!',RuntimeWarning)
    self.y=convolution.convolve(self.y,kernel,*args,**kwargs)
    self.convolved=True
  def gconvolve(self,fwhm):
    """ Convolve spectrum using a gaussian of given fwhm (in units of x axis) """
    gkernel=convolution.Gaussian1DKernel(fwhm)
    self.convolve(gkernel)
  def smooth(self,window_len=11,window='hanning'):
    """
    smooth(window_len=11,window='hanning')

    Smooth the spectrum using the given window of requested type and size.
    The supported smoothing functions are: Bartlett, Blackman, Hanning,
    Hamming, and flat (i.e. moving average).
    This method has been adapted from http://stackoverflow.com/a/5516430

    Parameters
    ----------
    window_len : int, optional
      Requested window size, in increments of x axis.
    window : string, optional
      Requested window type. Possible values are: 'flat', 'hanning',
      'hamming', 'bartlett', and 'blackman'.
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
      cAx.plot(self.x,self.y,'k-',drawstyle='steps-mid')
      cBaseliner = utils.Baseliner(cAx,self)
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
        iBaseline=np.logical_and(iBaseline,np.logical_or(np.less(self.x.value,cWindow[0]),np.greater(self.x.value,cWindow[1])))
      else:
        iBaseline=np.logical_or(iBaseline,np.logical_and(np.greater(self.x.value,cWindow[0]),np.less(self.x.value,cWindow[1])))
    baseline = np.polyfit(self.x.value[iBaseline],self.y.value[iBaseline],degree)
    if not(np.all(np.isfinite(baseline))):
      raise RuntimeError('Baseline is non-finite!')
    fixedY = self.y.value
    for cPower in range(degree+1):
      fixedY=fixedY-baseline[degree-cPower]*self.x.value**cPower
    self.y=fixedY*self.y.unit
    if self.dy is None:
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
    iCheckrange=np.ones_like(self.y.value,dtype=bool)
    if np.any(checkrange):
      minX=checkrange[0]
      maxX=checkrange[1]     
      iCheckrange=np.logical_and(iCheckrange,np.logical_and(
                        np.less_equal(minX,self.x.value),np.greater_equal(maxX,self.x.value)))
    return np.nanmax(self.y[iCheckrange])
  def min(self,checkrange=None):
    """
    Returns minimum y of the spectrum.
    If checkrange is set, returns maximum inside of that range.
    """
    iCheckrange=np.ones_like(self.y.value,dtype=bool)
    if np.any(checkrange):
      minX=checkrange[0]
      maxX=checkrange[1]     
      iCheckrange=np.logical_and(iCheckrange,np.logical_and(
                        np.less_equal(minX,self.x.value),np.greater_equal(maxX,self.x.value)))
    return np.nanmin(self.y[iCheckrange])
  def info(self):
    print '---'
    print 'Summary for spectrum '+self.name
    print 'x unit: '+str(self.x.unit)
    print 'min(x): '+str(np.nanmin(self.x.value))
    print 'max(x): '+str(np.nanmax(self.x.value))
    print 'y unit: '+str(self.y.unit)
    print 'min(y): '+str(np.nanmin(self.y.value))
    print 'max(y): '+str(np.nanmax(self.y.value))
    print 'baselined: '+str(self.baselined)
    print 'convolved: '+str(self.convolved)
    print '---'

class AbsorptionSpectrum(BaseSpectrum):
  """
  An absorption spectrum, with all the
  specific details that involves.
  Units on the y axis are in optical depth
  """
  def __init__(self,iWn,iOd,specname='Unknown absorption spectrum',nondata=[]):
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
    BaseSpectrum.__init__(self,self.wn,self.od,specname=specname,nondata=nondata)
  def plotod(self,iAx,in_wl=False,*args,**kwargs):
    """
    Plot the optical depth spectrum as function of wavenumber
    to given axis unless flag is set.
    """
    if in_wl:
      self.plot(iAx,x='wl',y='od',*args,**kwargs)      
    else:
      self.plot(iAx,x='wn',y='od',*args,**kwargs)


class LabSpectrum(AbsorptionSpectrum):
  """
  Laboratory spectrum class from optical constants.
  Inherits AbsorptionSpectrum class propertries.
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
    AbsorptionSpectrum.__init__(self,iWn*u.kayser,self.cabs_vol*utils.unit_od,specname=specname)
  def plotnk(self,ax1,ax2,*args,**kwargs):
    """
    Plot the optical constants as function of wavenumber to the two matplotlib axes given.
    Accepts all the same args and kwargs as pyplot.plt
    """
    ax1.plot(self.wn,self.n,*args,**kwargs)
    ax2.plot(self.wn,self.k,*args,**kwargs)