# -*- coding: utf-8 -*-
#Omnifit - beta2
#By Aleksi Suutarinen Sept 2013
#---------------
#Library imports
#---------------
import matplotlib
import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import leastsq
from lmfit import minimize, Parameters, Parameter
import re
import warnings
import time
import pickle
import os
from bhcoat import *
#----------------------
#Main class definitions
#----------------------
class spectrum:
  """ Generic 1d spectrum class. """
  def __init__(self,iX,iY,specname='Unknown spectrum',nonData=[],xUnit='Unknown',yUnit='Unknown'):
    """
    Init requires input of x axis values (wavelength etc.) and y axis values (intensity etc.)
    nonData adds extra variable names to the list of variables to be ignored by
    the dropbad function.
    """
    if len(iX) != len(iY):                                  #Check that input is sane
      raise Exception('Input arrays have different sizes.')
    self.x=np.array(iX,dtype='float64')                     #X axis
    self.y=np.array(iY,dtype='float64')                     #Y axis
    self.dy=np.nan                                          #Do a baseline to get this
    self.name=str(specname)                                 #Spectrum name
    self.baselined=False                                    #Has the spectrum been baselined?
    self.convolved=False                                    #Has the spectrum been convolved?
    self.drawstyle='steps-mid'
    self.xUnit=xUnit
    self.yUnit=yUnit
    self.nonData = [
                      'nonData',\
                      'name',\
                      'convolved','baselined',\
                      'drawstyle',\
                      'xUnit','yUnit','dy'\
                   ]
    for cNonData in nonData:                                #Add the extra non-array variable names into nonData
      if not cNonData in self.nonData:
        self.nonData.append(cNonData)                       
    self.dropbad()                                          #Drop bad data.
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
  def dropbad(self):
    """
    Make spectrum go through its own vars and drop all the bad ones.
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
        raise Exception('Anomalous variable length detected in spectrum!')
      iGoodones = np.logical_and(iGoodones,np.isfinite(cVar))
    for cVarname in ownvarnames:
      self.__dict__[cVarname]=self.__dict__[cVarname][iGoodones]
  def plot(self,iAx,plotstyle='k-',drawstyle=None,x=None,y=None):
    """
    Plot x,y into given MPL axis.
    They can be overridden with different
    variable names.
    """
    if x:
      cX = self.__dict__[x]
    else:
      cX = self.x
    if y:
      cY = self.__dict__[y]
    else:
      cY = self.y
    if len(cY) != len(cY):
      raise Exception('Plottable x and y values of different length!')
    if drawstyle:
      cDrawstyle=drawstyle
    else:
      cDrawstyle=self.drawstyle
    iAx.plot(cX,cY,plotstyle,drawstyle=cDrawstyle)
  def wl2wn(self):
    """
    Convert x axis units from wavelength [um]
    to wavenumer [cm^-1]
    """
    if self.xUnit != 'Wavelength':
      raise Exception('x axis units are not of wavelength, they are '+self.xUnit)
    self.x=wl2wn(self.x)
    self.xUnit = 'Wave number'
    self.sort()
  def wn2wl(self):
    """
    Convert x axis units from wavenumer [cm^-1]
    to wavelength [um]
    """
    if self.xUnit != 'Wave number':
      raise Exception('x axis units are not of wave number, they are '+self.xUnit)
    self.x=wn2wl(self.x)
    self.xUnit = 'Wavelength'
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
    newSpec = spectrum(newX,newY,specname=self.name+'(cropped)',
                                 nonData=self.nonData,
                                 xUnit=self.xUnit,
                                 yUnit=self.yUnit)
    newSpec.baselined = self.baselined
    newSpec.convolved = self.convolved
    newSpec.drawstyle = self.drawstyle
    for cVarName in self.__dict__.keys():
      if not cVarName in newSpec.__dict__.keys():
        newSpec.__dict__[cVarName]=self.__dict__[cVarName]
    return newSpec
  def interpolate(self,targSpectrum):
    """
    Interpolate spectrum to match target spectrum resolution.
    Does not modify current spectrum. Returns a new one.
    New spectrum metadata is taken from target spectrum.
    """
    if self.xUnit != targSpectrum.xUnit:
      raise Exception('Spectrums have different units on x axis!')
    if self.yUnit != targSpectrum.yUnit:
      raise Exception('Spectrums have different units on y axis!')
    newX=targSpectrum.x
    newY=np.interp(newX,self.x,self.y)
    newSpec=spectrum(newX,newY,
                    specname=self.name+'(interpolated: '+targSpectrum.name+')',
                    nonData=self.nonData,
                    xUnit=self.xUnit,yUnit=self.yUnit)
    newSpec.baselined = targSpectrum.baselined
    newSpec.convolved = targSpectrum.convolved
    newSpec.drawstyle = targSpectrum.drawstyle
    for cVarName in targSpectrum.__dict__.keys():
      if not cVarName in newSpec.__dict__.keys():
        newSpec.__dict__[cVarName]=targSpectrum.__dict__[cVarName]
    return newSpec
  def yat(self,x):
    """ Return value of y at x """
    return np.interp(x,self.x,self.y)
  def convolve(self,psf):
    """ Convolve spectrum y with given psf """
    if self.convolved:
      warnings.warn('Spectrum '+self.name+' has already been convolved once!',RuntimeWarning)
    #if not(self.baselined):
      #warnings.warn('Spectrum '+self.name+' has not been baselined yet. Recommend baselining before convolution.',RuntimeWarning)
    self.y=np.convolve(self.y,psf,mode='same')
    self.convolved=True
  def gpsf(self,fwhm):
    """ Return PSF for a gaussian convlution function of given fwhm (in units of x axis """
    deltaX=np.mean(np.diff(self.x))
    tempX=np.arange(-10*fwhm,10*fwhm+0.001*deltaX,deltaX)
    if len(tempX)>len(self.x):
      raise Exception('Length of convolving array must be less than convolved array for this to work. Decrease the fwhm.')
    gPsf=1.0*np.exp(-2.35*(tempX)**2./fwhm**2.)
    return gPsf
  def autopsf(self):
    """ Return automatically generated psf in the form of a gaussian with a FWHM of average x step amount """
    return self.gpsf(np.mean(np.diff(self.x)))
  def gconvolve(self,fwhm):
    """ Convolve spectrum using a gaussian of given fwhm (in units of x axis) """
    gPsf=self.gpsf(fwhm)
    #gPsf=gaussian(np.arange(-10*fwhm,10*fwhm,deltaX),[1.0,0.0,fwhm])
    self.convolve(gPsf)
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
    y=numpy.convolve(w/w.sum(),s,mode='same')
  def smooth(self):
    """ Convolve spectrum with a gaussian by 50% """
    deltaX=np.mean(np.diff(self.x))
    
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
      plt.show(cFig)
      windows=cBaseliner.windows
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
      raise Exception('Baseline is non-finite!')
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
    iCheckrange=np.isfinite(self.y)
    if np.any(checkrange):
      minX=checkrange[0]
      maxX=checkrange[1]     
      iCheckrange=np.logical_and(iCheckrange,np.logical_and(
                        np.less_equal(minX,self.x),np.greater_equal(maxX,self.x)))
    return np.max(self.y[iCheckrange])
  def min(self,checkrange=None):
    """
    Returns minimum y of the spectrum.
    If checkrange is set, returns maximum inside of that range.
    """
    iCheckrange=np.isfinite(self.y)
    if np.any(checkrange):
      minX=checkrange[0]
      maxX=checkrange[1]     
      iCheckrange=np.logical_and(iCheckrange,np.logical_and(
                        np.less_equal(minX,self.x),np.greater_equal(maxX,self.x)))
    return np.min(self.y[iCheckrange])
  def info(self):
    print '---'
    print 'Summary for spectrum '+self.name
    print 'x unit: '+self.xUnit
    print 'min(x): '+np.min(self.x)
    print 'max(x): '+np.max(self.x)
    print 'y unit: '+self.yUnit
    print 'min(y): '+np.min(self.y)
    print 'max(y): '+np.max(self.y)
    print 'baseline: '+str(self.baselined)
    print 'convolved: '+str(self.convolved)
    print '---'

class absorptionSpectrum(spectrum):
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
    self.od = np.array(iOd,dtype='float64') #Optical depth
    self.wn = np.array(iWn,dtype='float64') #Wave number
    self.wl = wn2wl(self.wn)
    spectrum.__init__(self,self.wn,self.od,specname=specname,xUnit='Wave number',yUnit='Optical depth',nonData=nonData)
  def plotod(self,iAx,in_wl=False,**kwargs):
    """
    Plot the optical depth spectrum as function of wavenumber
    to given axis unless flag is set.
    """
    if in_wl:
      self.plot(iAx,x='wl',y='od',**kwargs)      
    else:
      self.plot(iAx,x='wn',y='od',**kwargs)


#class labSpectrum(absorptionSpectrum):
  #"""
  #Laboratory spectrum class.
  #Inherits absorptionspectrum class propertries.
  #"""
  #def __init__(self,iWn,iOd,specname='Unknown laboratory spectrum'):
    #"""
    #Init requires input of wavenumber array [cm^-1] and od array.
    #Optional input: Name of spectrum
    #"""
    #absorptionSpectrum.__init__(self,iWn,iOd,specname=specname)

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
    if len(iWn) != len(iN):
      raise TypeError('Input arrays have different sizes.')
    self.cabs,self.cabs_vol,self.cscat_vol,self.ctot=cde_correct(iWn,iN,iK)
    self.n=np.array(iN,dtype='float64')
    self.k=np.array(iK,dtype='float64')
    absorptionSpectrum.__init__(self,iWn,self.cabs_vol,specname=specname)
  def plotnk(self,xrange=None):
    """ Plot the optical constants as function of wavenumber """
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
    plt.show()
    plt.close()

class fitter():
  """
  The omnifitter class.
  The class consists of two major components:
    -The target data points (X and Y; X is assumed same for all functions)
    -The function list, consisting of either lab data or theoretical functions
  """
  def __init__(self,iX,iY,dY=1.0,modelname='Unknown model',psf=None,fitrange=None,color='blue',customfunctions=False):
    """ Initialise with target spectrum that the rest will be fitted to. """
    if len(iX) != len(iY):
      raise Exception('Input arrays have different sizes.')
    self.fitrange=fitrange
    self.targX=iX
    self.targY=iY
    self.targdY=dY
    self.modelname=modelname
    self.color=color
    self.psf=psf
    self.customfunctions=customfunctions
    self.funcList=[]
  def add_lab(self,iSpectrum,iParams,funcname=None,color='red'):
    """
    Add laboratory spectrum to the fitting pool.
    Spectrum must be interpolated and convolved to match with target.
    Uses lmfit params as the parameter list
    """
    if not(funcname):
      funcname=iSpectrum.name
    if len(iSpectrum.x) != len(self.targX):
      raise Exception('Input spectrum has wrong size!')
    self.funcList.append({'type':'lab','shape':iSpectrum.y,'params':iParams,'name':funcname,'color':color})
  def add_theory(self,iShape,iParams,funcname='Unknown function',color='red'):
    """
    Add theoretical function to fitting pool.
    Supported functions:
      gaussian, lorentzian, omni_custom
      (omni_custom needs to be defined separately)
    Uses lmfit params as the parameter list
    """
    supported_functions=['gaussian','lorentzian','flipped_egh']
    if iShape not in supported_functions and not self.customfunctions:
       raise Exception('Function shape not supported. Supported function shapes: '+str(supported_functions))
    self.funcList.append({'type':'theory','shape':iShape,'params':iParams,'name':funcname,'color':color})

  def perform_fit(self):
    """
    Perform least-squares fitting to the function list
    """
    self.fitPars = self.extract_pars()
    self.fitRes=minimize(self.fit_residual,self.fitPars,epsfcn=0.05)
    if not(self.fitRes.success):
      raise Exception('Fitting failed!')
  def fit_residual(self,iPar,custrange=None):
    """ Calculate residual of all the functions compared to the target function """
    if custrange==None:
      fitrange=self.fitrange
    else:
      fitrange=custrange
    residual=1.0*self.targY
    totModel=np.zeros(len(residual))
    for indFunc,cFunc in enumerate(self.funcList):
      oPar=Parameters()
      cParlist = cFunc['params']
      for cPar in cParlist.values():
        ciPar=iPar[self.func_ident(indFunc)+cPar.name]
        oPar.add(cPar.name,
                 value=ciPar.value,vary=ciPar.vary,
                 min=ciPar.min,max=ciPar.max,
                 expr=ciPar.expr)
      residual-=self.parse_function(oPar,cFunc)
    #Crop out not-numbers and fitting range exterior if necessary
    if np.any(fitrange):
      fitInd=np.isinf(residual)
      for cRange in fitrange:
        fitInd=np.logical_or(fitInd,np.logical_and(
               np.less_equal(cRange[0],self.targX),
               np.greater_equal(cRange[1],self.targX)))
    else:
      fitInd=np.isfinite(residual)
    return residual[fitInd]
  def chisq(self,checkrange=None):
    """ 
    Return chi squared of fit, either in a custom range 
    or in the range used by the fit
    """
    residual = self.fit_residual(self.fitPars,custrange=checkrange)
    return np.sum((residual**2.0)/(self.targdY**2.0))
  def plot_fitresults(self,iAx,autorange=True,drawstyle='steps-mid',legend=True):
    """ Plot the fitting results to given axis """
    if autorange:
      iAx.set_xlim(np.min(self.targX),np.max(self.targX))
    iAx.plot(self.targX,self.targY,'k-',lw=1,drawstyle=drawstyle)
    legList = [self.modelname]
    #totres=self.targ_y+self.fitres.residual
    totRes=np.zeros(len(self.targY))
    for indFunc,cFunc in enumerate(self.funcList):
      oPar=Parameters()
      cParList = cFunc['params']
      cCol = cFunc['color']
      for cPar in cParList.values():
        cFitPar=self.fitPars[self.func_ident(indFunc)+cPar.name]
        oPar.add(cPar.name,
                 value=cFitPar.value,vary=cFitPar.vary,
                 min=cFitPar.min,max=cFitPar.max,
                 expr=cFitPar.expr)
      funcRes = self.parse_function(oPar,cFunc)
      totRes+=funcRes
      iAx.plot(self.targX,funcRes,lw=2,drawstyle=drawstyle,color=cCol)
      legList.append(cFunc['name'])
    legList.append('Total fit')
    iAx.plot(self.targX,totRes,lw=3,drawstyle=drawstyle,color=self.color)
    if legend:
      iAx.legend(legList,shadow=True)
  def fitresults_tofile(self,filename):
    """
    Export fit results to two output files.
    First file is filename.xml, which contains information about both
    the best-fit parameters and of function names etc.
    Second file is filename.csv, which contains x and y data of the fitted
    models, as would be visualized in a plotted fit result.
    First column of csv is the x value, which is shared by all models.
    Second column is y value of data that was being fitted to.
    Third column is total sum of fitted models.
    Fourth to Nth columns are the individual models, in the order
    described in filename.xml.
    """
    filename_csv = filename+'.csv'
    filename_xml = filename+'.xml'
    file_xml = open(filename_xml,'w')
    file_xml.write('<!-- Automatically generated information file for csv file '+filename_csv+'-->\n')
    file_xml.write('<INFO file="'+filename_csv+'">\n')
    file_xml.write('<MODELNAME>'+self.modelname+'</MODELNAME>\n')
    file_xml.write('<HAVEPSF>'+str(self.psf != None)+'</HAVEPSF>\n')
    file_xml.write('<RMS_DATA>'+str(self.targdY)+'</RMS_DATA>\n')
    file_xml.write('<NUMBER_FUNCTIONS>'+str(len(self.funcList))+'</NUMBER_FUNCTIONS>\n')
    outdata_csv = np.vstack([self.targX,self.targY])
    outdata_functions = np.empty([0,len(self.targX)])
    totRes = np.zeros(len(self.targX))
    for indFunc,cFunc in enumerate(self.funcList):
      file_xml.write('<FUNCTION name="'+cFunc['name']+'">\n')
      file_xml.write('<TYPE>')
      if cFunc['type'] == 'theory':
        file_xml.write(cFunc['shape'])
      elif cFunc['type'] == 'lab':
        file_xml.write('lab')
      else:
        file_xml.write('unknown'+'\n')
      file_xml.write('</TYPE>\n')
      file_xml.write('<DETECTION>'+str(not self.lowsigma(sigma=2.0)[cFunc['name']])+'</DETECTION>\n')
      file_xml.write('<CSV_COLUMN>'+str(indFunc+3)+'</CSV_COLUMN>\n')
      cParlist = cFunc['params']
      file_xml.write('<NUMBER_PARAMS>'+str(len(cParlist))+'</NUMBER_PARAMS>\n')
      oPar=Parameters()
      for cPar in cParlist.values():
        file_xml.write('<PARAMETER name="'+cPar.name+'">\n')
        cFitPar=self.fitPars[self.func_ident(indFunc)+cPar.name]
        oPar.add(cPar.name,
                 value=cFitPar.value,vary=cFitPar.vary,
                 min=cFitPar.min,max=cFitPar.max,
                 expr=cFitPar.expr)
        file_xml.write('<VALUE>'+str(cFitPar.value)+'</VALUE>\n')
        file_xml.write('</PARAMETER>\n')
      funcRes = self.parse_function(oPar,cFunc)
      outdata_functions = np.vstack([outdata_functions,funcRes])
      totRes += funcRes
      file_xml.write('</FUNCTION>\n')
    file_xml.write('</INFO>')
    file_xml.close()
    outdata_csv = np.vstack([outdata_csv,totRes,outdata_functions])
    np.savetxt(filename_csv,outdata_csv.transpose(),delimiter=',',header='For info, see '+filename_xml)
  def lowsigma(self,sigma=1.0):
    """
    Return dictionary with boolean values indicating
    whether the fitting results are below a certain
    sigma multiple value
    """
    minY = sigma*self.targdY
    out = {}
    totRes = np.zeros(len(self.targX))
    for indFunc,cFunc in enumerate(self.funcList):
      cParlist = cFunc['params']
      oPar=Parameters()
      for cPar in cParlist.values():
        cFitPar=self.fitPars[self.func_ident(indFunc)+cPar.name]
        oPar.add(cPar.name,
                 value=cFitPar.value,vary=cFitPar.vary,
                 min=cFitPar.min,max=cFitPar.max,
                 expr=cFitPar.expr)
      funcRes = self.parse_function(oPar,cFunc)
      if np.max(funcRes) < minY:
        out[cFunc['name']] = True
      else:
        out[cFunc['name']] = False
      totRes += funcRes
    if np.max(totRes) < minY:
      out['total'] = True
    else:
      out['total'] = False
    return out
  def fit_results(self):
    """ Return all fitting results as a dictionary"""
    oResults={}
    for indFunc,cFunc in enumerate(self.funcList):
      oKeyname_base=cFunc['name']
      oKeyind=0
      oKeyname=oKeyname_base
      while oResults.__contains__(oKeyname): #In case of duplicate function names
        oKeyind+=1
        oKeyname=oKeyname_base+'(duplicate '+str(oKeyind)+')'
      oResults[cFunc['name']]=self.fit_result(indFunc)
    return oResults
  def fit_result(self,indFunc):
    """ Return fitting results for a specific function """
    oParlist=self.funcList[indFunc]['params']
    for cParname in oParlist.keys():
      coPar=self.fitPars[self.func_ident(indFunc)+cParname]
      coPar.name=cParname
      oParlist[cParname]=coPar
    return oParlist
  def parse_function(self,iPar,iFunc):
    """ Parse the input function, insert parameters, return result """
    if iFunc['type']=='lab':
      #shiftData=np.interp(self.targX+iPar['shift'].value,self.targX,iFunc['shape'])
      # shiftamount = int(np.round(iPar['shift'].value))
      funcRes=muldata(iFunc['shape'],iPar['mul'].value)
      # leftEdge = funcRes[0]
      # rightEdge = funcRes[-1]
      # if shiftamount != 0:
      #   funcRes=np.roll(funcRes,shiftamount)
      #   if shiftamount >= 0:
      #     funcRes[:shiftamount] = leftEdge
      #   else:
      #     funcRes[shiftamount:] = rightEdge
      #funcRes=muldata(iFunc['shape'],iPar['mul'].value)
    elif iFunc['type']=='scatter':
      funcres=scatterdata(self.targX,iFunc['shape'],iFunc['grain'],iPar,self.psf)
    elif iFunc['type']=='theory':
      funcRes=globals()[iFunc['shape']](self.targX,iPar,self.psf)
    else:
      raise Exception('Unknown function type!')
    return funcRes
  def extract_pars(self):
    """
    Extract the paramers from the function list so they can be manipulated by the residual minimization routines.
    """
    oPars=Parameters()
    for indFunc,cFunc in enumerate(self.funcList):
      cParlist = cFunc['params']
      for cPar in cParlist.values():
        oPars.add(self.func_ident(indFunc)+cPar.name,
                  value=cPar.value,vary=cPar.vary,
                  min=cPar.min,max=cPar.max,
                  expr=cPar.expr)
    return oPars
  def func_ident(self,indFunc):
    """ 
    Return function identifier string.
    Used with function fitting
    """
    return '__Func'+str(indFunc)+'__'
#-------------------------
#Support class definitions
#-------------------------
class baseliner:
  """
  baseliner of the objects
  """
  def __init__(self,ax,spec,power=1):
    self.ax = ax
    self.spec = spec
    self.x = spec.x
    self.y = spec.y
    self.power = power
    self.limlo=None
    self.limhi=None
    self.minx=np.min(self.x)
    self.maxx=np.max(self.x)
    self.miny=np.min(self.y)
    self.maxy=np.max(self.y)
    self.ax.set_xlim(self.minx,self.maxx)
    self.ax.set_ylim(self.miny,self.maxy)
    self.specplot,=self.ax.plot(self.x,self.y,'k-',drawstyle=spec.drawstyle)
    self.buttonListener = self.ax.figure.canvas.mpl_connect('button_press_event', self.mouse_press)
    self.keyListener = self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)
    self.windows=[]
  def key_press(self, event):
    if event.key=='q':
      self.skip()
    if event.key=='a' and self.limlo != None and self.limhi != None:
      self.addwindow(self.limlo,self.limhi)
      self.ax.plot([self.limlo,self.limlo],[self.miny,self.maxy],'g-')
      self.ax.plot([self.limhi,self.limhi],[self.miny,self.maxy],'g-')
      self.remlim()
      self.remlim()
      print 'Window added. Ready to receive another one.'
    else:
      return
  def mouse_press(self, event):
    if event.button==1:
      self.setlim(event.xdata)
    elif event.button==2:
      return
    elif event.button==3:
      self.remlim()
  def skip(self):
    plt.close()
  def setlim(self,i_x):
    if self.limlo==None:
      self.limlo=i_x
      self.limloplot,=self.ax.plot([i_x,i_x],[self.miny,self.maxy],'b-')
      self.ax.figure.canvas.draw()
    elif self.limhi==None:
      self.limhi=i_x
      self.limhiplot,=self.ax.plot([i_x,i_x],[self.miny,self.maxy],'b-')
      self.ax.figure.canvas.draw()
      print 'Ready for finalising. Press once more to do so, or press a to add another window.'
    else:
      self.finalise()
  def remlim(self):
    if self.limhi!=None:
      self.limhi=None
      self.limhiplot.set_ydata([self.miny,self.miny])
      self.ax.figure.canvas.draw()
    elif self.limlo!=None:
      self.limlo=None
      self.limloplot.set_ydata([self.miny,self.miny])
      self.ax.figure.canvas.draw()
    else:
      print 'No limits to cancel.'
  def addwindow(self,limlo,limhi):
    if limhi < limlo:
      limlo,limhi = limhi,limlo
    self.windows.append([limlo,limhi])
  def finalise(self):
    self.addwindow(self.limlo,self.limhi)
    self.ax.figure.canvas.mpl_disconnect(self.buttonListener)
    self.ax.figure.canvas.mpl_disconnect(self.keyListener)
    plt.close(self.ax.figure)

#--------------------------
#Misc. function definitions
#--------------------------
def wl2wn(iWavelength):
  """
  Convert wavelength [um] to wavenumber [cm^-1]
  """
  return 1.0e4/iWavelength

def wn2wl(iWavenumber):
  """
  Convert wavenumber [cm^-1] to wavelength [um]
  """
  return 1.0e4/iWavenumber

def cde_correct(wn,n,k):
  """
  CDE correction to  n,k data
  """
  wl=wn2wl(wn)
  m=np.vectorize(complex)(n,k)
  cabs_vol=np.empty(0)
  cabs=np.empty(0)
  cscat_vol=np.empty(0)
  for c_wn,c_wl,c_m in zip(wn,wl,m):
    m2=c_m**2.0
    im_part=((m2/(m2-1))*np.log(m2)).imag
    cabs_vol=np.hstack([cabs_vol,2.*(2.*np.pi/c_wl)*im_part])
    t_cabs=c_wn*(2.0*c_m.imag/(c_m.imag-1))*np.log10(c_m.imag)
    cabs=np.hstack([cabs,t_cabs])
    cscat_vol=np.hstack([cscat_vol,(c_wn**3.0/(6*np.pi))*t_cabs])
  ctot=cabs+cscat_vol
  return cabs,cabs_vol,cscat_vol,ctot
#----------------------------
#Fitting function definitions
#----------------------------
def scatterdata(wavenum,lab,ref_core,par,psf=None):
  """
  Scatter lab data with bhcoat
  Assumes x axis is wavenumber
  """
  radius_core=par['radius_core'].value
  radius_mantle=par['radius_mantle'].value
  wavelen=wn2wl(wavenum)*1.0e-6
  out_y=np.empty(0)
  for c_wavelen in wavelen:
    c_n=np.interp(c_wavelen,lab.wl,lab.n)
    c_k=np.interp(c_wavelen,lab.wl,lab.k)
    ref_mantle=np.complex(c_n,c_k)
    #Qext=bhcoat(c_wavelen,radius_core,radius_mantle,ref_core,ref_mantle)['Qext']
    Qsca=bhcoat(c_wavelen,radius_core,radius_mantle,ref_core,ref_mantle)['Qsca']
    Csca=Qsca*np.pi*radius_mantle**2.0
    out_y = np.hstack((out_y,Csca))#3*Qsca/(4*radius_mantle)))
  c_fig=plt.figure()
  c_axis = c_fig.add_subplot(111)
  c_axis.plot(wavenum,out_y,'k-',drawstyle=drawstyle)
  plt.show()
  plt.close()
  if not(np.any(psf)):
    return out_y
  else:
    return np.convolve(out_y,psf,mode='same')
def muldata(data,mul):#par):
  """ Multiply data with par and return result """
  return mul*data

def flipped_egh(x,par,psf):
  """
  Flipped EGH (exponential-gaussian hybrid)
  For normal EGH, see:
  Lan & Jorgenson, Journal of Chromatography A, 915 (2001) 1-13
  Parameters:
  H = magnitude of peak maximum
  xR = retention time
  w = std. dev of precursor gaussian
  tau = time constant of precursor exponential
  """
  H=par['H'].value
  xR=par['xR'].value
  w=par['w'].value
  tau=par['tau'].value
  expFactor = np.exp((-1.0*(xR-x)**2.0)/(2.0*w*w+tau*(xR-x)))
  out_y = np.where(2.0*w*w+tau*(xR-x)>0,H*expFactor,0.0)
  if not(np.any(psf)):
    return out_y
  else:
    return np.convolve(out_y,psf,mode='same')

def poly3(x,par,psf=None):
  """ A 3rd order polynomial function """
  p1=par['par1'].value
  p2=par['par2'].value
  p3=par['par3'].value
  p4=par['par4'].value
  out_y=p1+p2*x+p3*x*x+p4*x*x*x
  if not(np.any(psf)):
    return out_y
  else:
    return np.convolve(out_y,psf,mode='same')

def gaussian(x,par,psf=None):
  """ A gaussian function """
  peak=par['peak'].value
  fwhm=par['fwhm'].value
  pos=par['pos'].value
  out_y=peak*np.exp(-2.35*(x-pos)**2./fwhm**2.)
  if not(np.any(psf)):
    return out_y
  else:
    return np.convolve(out_y,psf,mode='same')
def lorentzian(x,par,psf=None):
  """ A lorentzian function """
  lor1=par['lor1'].value
  lor2=par['lor2'].value
  lor3=par['lor3'].value
  peak=par['peak'].value
  pos=par['pos'].value
  lorentz_oscillator=lor1+lor2**2./(pos**2.-x**2.-lor3*x*1.j)
  out_y=peak*x*np.imag(2.*lorentz_oscillator*np.log10(lorentz_oscillator)/(lorentz_oscillator-1.))
  if not(np.any(psf)):
    return out_y
  else:
    return np.convolve(out_y,psf,mode='same')
