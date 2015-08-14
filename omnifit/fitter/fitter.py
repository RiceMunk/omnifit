import numpy as np
from lmfit import minimize, Parameters, Parameter
from astropy import units as u
from .. import spectrum
from functions import *

class Fitter():
  """
  A class for multi-component fitting to spectroscopic data of ices.

  This is the heart of Omnifit, which receives spectra from the spectrum
  module, and is then capable of fitting an arbitrary number of different
  components to the target spectrum thus designated.

  Attributes
  ----------
  target_x : numpy.ndarray
    The x axis of the target spectrum, e.g. the wavelength.
  target_y : numpy.ndarray
    The y axis of the target spectrum, e.g. the optical depth.
  target_dy : float
    A single number expressing the average uncertainty of the y
    axis data.
  modelname : string
    A human-readable name for the model being fitted.
  psf : Nonetype or numpy.ndarray or astropy.convolution.Kernel
    If set, this attribute can be used to give a kernel which should
    be used to convolve all the fitted data with.
  fitrange : Nonetype or list
    If set, this specifies the inclusive limits to which
    the fitting should be performed in x axis coordinates.
    For example a fitrange of [[200,250],[300,350]] sets
    two fitting windows of 200 to 250, and 300 to 350.
  color : string
    A string inidcating the desired plotting color of the target
    data, in a format understandable by matplotlib.
  funclist : list
    A list containing all the fittable functions. Each list entry
    is a dictionary containing the following keys and values:
      * 'name' : A human-readable name for the function being fitted,
        in string format.
      * 'color' : A string inidcating the desired plotting color of
        the data, in a format understandable by matplotlib.
      * 'type' : A string indicating what type of data the function
        consists of. It can be either 'analytical' or 'empirical',
        indicating an analytical function or empirical spectrum,
        respectively.
      * 'shape' : The shape of the function being fitted. In the case
        of an analytical function, this is a string indicating the
        callable name of the function. In the case of an empirical
        spectrum, this is the y-axis data from the spectrum.
      * 'params' : an lmfit.parameter instance containing the fitting
        parameters appropriate to the data being fitted.
  fitpars : lmfit.parameter.Parameters
    This is where the fitting parameters are stored during and after
    minimization.
  fitres : lmfit.minimizer.Minimizer
    The fitting results are stored in this class, as documented in
    lmfit.
  """
  def __init__(self,x,y,dy=1.0,modelname='Unknown model',psf=None,fitrange=None,color='black'):
    """
    Fitter(x,y,dy=1.0,modelname='Unknown model',psf=None,fitrange=None,color='black')

    Constructor for the Fitter class. Initialisation happens by
    designating the target spectrum.

    Parameters
    ----------
    x : numpy.ndarray
      The x axis of the target spectrum, e.g. the wavelength.
    y : numpy.ndarray
      The y axis of the target spectrum, e.g. the optical depth.
    dy : float
      A single number expressing the average uncertainty of the y
      axis data.
    modelname : string
      A human-readable name for the model being fitted.
    psf : Nonetype or numpy.ndarray or astropy.convolution.Kernel
      This attribute can be used to give a kernel which should be
      used to convolve all the fitted data with.
    fitrange : Nonetype or list
      If set, this specifies the inclusive limits to which
      the fitting should be performed in x axis coordinates.
      For example a fitrange of [[200,250],[300,350]] sets
      two fitting windows of 200 to 250, and 300 to 350.
    color : string
      A string inidcating the desired plotting color of the target
      data, in a format understandable by matplotlib.
    """
    if len(x) != len(y):
      raise RuntimeError('Input arrays have different sizes.')
    self.target_x=x
    self.target_y=y
    self.target_dy=dy
    self.modelname=modelname
    self.psf=psf
    self.fitrange=fitrange
    self.color=color
    self.funclist=[]
  @classmethod
  def fromspectrum(cls,spectrum,**kwargs):
    """
    Fitter.fromspectrum(spectrum,**kwargs)

    An alternate way to initialise Fitter, by directly giving it
    a spectrum. Extracted data from the spectrum are the x, y,
    and (if the spectrum has been baselined) dy parameters.

    Attributes
    ----------
    spectrum : spectrum.BaseSpectrum
      The input spectrum.
    """
    if spectrum.baselined:
      return cls(spectrum.x.value,spectrum.y.value,spectrum.dy,**kwargs)
    else:
      return cls(spectrum.x.value,spectrum.y.value,**kwargs)
  def add_empirical(self,spectrum,params,funcname=None,color='red'):
    """
    add_empirical(spectrum,params,funcname=None,color='red')

    Add empirical data in the form of a spectrum to the fitting list.
    The spectrum must be interpolated to match the target x axis.

    Parameters
    ----------
    spectrum : spectrum.BaseSpectrum
      The input spectrum.
    params : lmfit.parameter.Parameters
      The input parameters. Specifically this must contain
      the 'mul' parameter, which indicates what value the
      spectrum will be multiplied with during fitting.
    funcname : Nonetype or string
      A human-readable name for the data being fitted.
      If this is left as None, the name of the spectrum will
      be used.
    color : string
      A string inidcating the desired plotting color of the
      data, in a format understandable by matplotlib.
    """
    if not(funcname):
      funcname=spectrum.name
    if not np.all(spectrum.x.value == self.target_x):
      raise RuntimeError('Input spectrum x axis does not match the target spectrum x axis.')
    self.funclist.append({'type':'empirical','shape':spectrum.y.value,'params':params,'name':funcname,'color':color})
  def add_analytical(self,shape,params,funcname='Unknown function',color='red'):
    """
    add_analytical(shape,params,funcname=None,color='red')

    Add analytical data in the form of a callable function to the
    fitting list.

    Parameters
    ----------
    shape : string
      The callable name of the function to be fitted.
    params : lmfit.parameter.Parameters
      The input parameters. These should be formatted in a way that
      the function defined by shape can understand them, and that
      function should be created in such a way that it can make use
      of lmfit parameters.
    funcname : string
      A human-readable name for the data being fitted.
    color : string
      A string inidcating the desired plotting color of the
      data, in a format understandable by matplotlib.
    """
    self.funclist.append({'type':'analytical','shape':shape,'params':params,'name':funcname,'color':color})

  def perform_fit(self,*args,**kwargs):
    """
    perform_fit(*args,**kwargs)

    Perform the least-squares fitting of all the functions in the
    function list to the target data.

    Parameters
    ----------
    *args and **kwargs are fed directly into lmfit.minimize.
    """
    self.fitpars = self.extract_pars()
    self.fitres=minimize(self.__fit_residual,self.fitpars,*args,**kwargs)
    if not(self.fitres.success):
      raise RuntimeError('Fitting failed!')
  def __fit_residual(self,params,custrange=None):
    """
    __fit_residual(params,custrange=None)

    This is an internal function used for calculating the total
    residual of the data against the fittings function(s), given
    a set of lmfit parameters. The residual calculation can also
    be limited to a specific x axis range.

    Parameters
    ----------
    params : lmfit.parameter.Parameters
      The parameters used for calculating the residual.
    custrange : Nonetype or list
      If set, this specifies the inclusive range within which
      the residual is calculated. Otherwise the fitting range
      specified during Initialisation is used.

    Returns
    -------
    The residual function within the fitting range with the given
    lmfit parameters.
    """
    if custrange==None:
      fitrange=self.fitrange
    else:
      fitrange=custrange
    residual=1.0*self.target_y
    totModel=np.zeros(len(residual))
    for indFunc,cFunc in enumerate(self.funclist):
      oPar=Parameters()
      cParlist = cFunc['params']
      for cPar in cParlist.values():
        cParams=params[self.func_ident(indFunc)+cPar.name]
        oPar.add(cPar.name,
                 value=cParams.value,vary=cParams.vary,
                 min=cParams.min,max=cParams.max,
                 expr=cParams.expr)
      residual-=self.parse_function(oPar,cFunc)
    #Crop out not-numbers and fitting range exterior if necessary
    if np.any(fitrange):
      fitInd=np.isinf(residual)
      for cRange in fitrange:
        fitInd=np.logical_or(fitInd,np.logical_and(
               np.less_equal(cRange[0],self.target_x),
               np.greater_equal(cRange[1],self.target_x)))
    else:
      fitInd=np.isfinite(residual)
    return residual[fitInd]
  def chisq(self,checkrange=None):
    """ 
    chisq(checkrange=None)

    Return chi squared of fit, either in a custom range 
    or in the range used by the fit.

    Parameters
    ----------
    checkrange : Nonetype or list
      If set, this specifies the inclusive range within which
      the chi squared value is calculated. Otherwise the fitting 
      range specified during Initialisation is used.

    Returns
    -------
    The chi squared within the desired ranged.
    """
    residual = self.__fit_residual(self.fitpars,custrange=checkrange)
    return np.sum((residual**2.0)/(self.target_dy**2.0))
  def plot_fitresults(self,ax,lw=[1,2,3],color_total='blue',legend=True,**kwargs):
    """
    plot_fitresults(ax,lw=[1,2,3],color_total='blue',legend=True,**kwargs)
    
    Plot the fitting results to the given matplotlib axis, with a
    number of optional parameters specifying how the different plottable
    components are presented.

    Parameters
    ----------
    axis : matplotlib.axis
      The axis which the plot will be generated in.
    lw : list
      This list of 3 numbers specifies the line widths of the target
      spectrum, the fitted functions, and the total fit, respectively.
    color_total : string
      A string inidcating the desired plotting color of the total sum
      of the fit results, in a format understandable by matplotlib.
      The colors of the target spectrum and the fitted functions are
      specified during their initialisation and addition.
    legend : bool
      If set to True, a legend is automatically created using the
      target spectrum and fitted function names.
    **kwargs can be used to pass additional plotting
      parameters to the matplotlib plotting routine, as documented
      in the matplotlib documentation.
    """
    if autorange:
      ax.set_xlim(np.min(self.target_x),np.max(self.target_x))
    ax.plot(self.target_x,self.target_y,color=self.color,lw=lw[0],**kwargs)
    legList = [self.modelname]
    #totres=self.targ_y+self.fitres.residual
    totRes=np.zeros(len(self.target_y))
    for indFunc,cFunc in enumerate(self.funclist):
      oPar=Parameters()
      cParList = cFunc['params']
      cCol = cFunc['color']
      for cPar in cParList.values():
        cFitPar=self.fitpars[self.func_ident(indFunc)+cPar.name]
        oPar.add(cPar.name,
                 value=cFitPar.value,vary=cFitPar.vary,
                 min=cFitPar.min,max=cFitPar.max,
                 expr=cFitPar.expr)
      funcRes = self.parse_function(oPar,cFunc)
      totRes+=funcRes
      ax.plot(self.target_x,funcRes,lw=lw[1],color=cCol,**kwargs)
      legList.append(cFunc['name'])
    legList.append('Total fit')
    ax.plot(self.target_x,totRes,lw=lw[2],color=color_total,**kwargs)
    if legend:
      ax.legend(legList,shadow=True)
  def fitresults_tofile(self,filename,detection_threshold=2.0):
    """
    fitresults_tofile(filename)

    Export fit results to two output files which are intended to be
    easily readable and paraseable with other software.

    The first file is filename.csv, which contains x and y data of
    the fitted models, as would be visualized in a plotted fit result.
    The first column of the csv is the x value, which is shared by all
    models.
    The second column is the y value of data that was being fitted to.
    The third column is total sum of fitted models.
    The fourth to Nth columns are the individual models, in the order
    described in the second file, filename.xml.

    The second file, filename.xml is an XML file containing additional
    information about the fitted data and the fit results which are not
    easily representable in a csv-formatted file. This data is
    formatted using the following XML elements:
      * INFO : Contains all the other elements described below, and has
        the attribute "file", which is the name of the csv file pair of
        this xml file.
      * MODELNAME : Contains the name of the model.
      * HAVEPSF : A boolean value indicating whether there is a PSF
        associated with the model.
      * RMS_DATA : The uncertainty of the data.
      * NUMBER_FUNCTIONS : An integer indicating how many functions
        have been fitted to the total data.
    In addition to the above elements, each fitted function has its own
    element, designated FUNCTION, having the attribute "name" which is
    the name of the function. FUNCTION contains the following elements:
      * TYPE : If the function is an empirical one, this contains the
        string "empirical". Otherwise it contains the name of the
        called analytical function.
      * DETECTION : When generating the contents of this element,
        The method lowsigma with the detection threshold designated
        by the parameter detection_threshold. The result given by
        the method is indicated here with a "True" or "False"
        depending on whether the result is considered a detection.


    Parameters
    ----------
      filename : string
        The extensionless version of the desired filename which the
        data should be exported to. As a result the files
        "filename.csv" and "filename.xml" are created.
      detection_threshold : float
        The threshold of detection to be used in determining whether
        the value contained by the DETECTION element is true or not.
    """
    filename_csv = filename+'.csv'
    filename_xml = filename+'.xml'
    file_xml = open(filename_xml,'w')
    file_xml.write('<!-- Automatically generated information file for csv file '+filename_csv+'-->\n')
    file_xml.write('<INFO file="'+filename_csv+'">\n')
    file_xml.write('<MODELNAME>'+self.modelname+'</MODELNAME>\n')
    file_xml.write('<HAVEPSF>'+str(self.psf != None)+'</HAVEPSF>\n')
    file_xml.write('<RMS_DATA>'+str(self.target_dy)+'</RMS_DATA>\n')
    file_xml.write('<NUMBER_FUNCTIONS>'+str(len(self.funclist))+'</NUMBER_FUNCTIONS>\n')
    outdata_csv = np.vstack([self.target_x,self.target_y])
    outdata_functions = np.empty([0,len(self.target_x)])
    totRes = np.zeros(len(self.target_x))
    for indFunc,cFunc in enumerate(self.funclist):
      file_xml.write('<FUNCTION name="'+cFunc['name']+'">\n')
      file_xml.write('<TYPE>')
      if cFunc['type'] == 'analytical':
        file_xml.write(cFunc['shape'])
      elif cFunc['type'] == 'empirical':
        file_xml.write('empirical')
      else:
        file_xml.write('unknown'+'\n')
      file_xml.write('</TYPE>\n')
      file_xml.write('<DETECTION>'+str(not self.lowsigma(sigma=detection_threshold)[cFunc['name']])+'</DETECTION>\n')
      file_xml.write('<CSV_COLUMN>'+str(indFunc+3)+'</CSV_COLUMN>\n')
      cParlist = cFunc['params']
      file_xml.write('<NUMBER_PARAMS>'+str(len(cParlist))+'</NUMBER_PARAMS>\n')
      oPar=Parameters()
      for cPar in cParlist.values():
        file_xml.write('<PARAMETER name="'+cPar.name+'">\n')
        cFitPar=self.fitpars[self.func_ident(indFunc)+cPar.name]
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
    minY = sigma*self.target_dy
    print self.target_dy
    out = {}
    totRes = np.zeros(len(self.target_x))
    for indFunc,cFunc in enumerate(self.funclist):
      cParlist = cFunc['params']
      oPar=Parameters()
      for cPar in cParlist.values():
        cFitPar=self.fitpars[self.func_ident(indFunc)+cPar.name]
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
    for indFunc,cFunc in enumerate(self.funclist):
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
    oParlist=self.funclist[indFunc]['params']
    for cParname in oParlist.keys():
      coPar=self.fitpars[self.func_ident(indFunc)+cParname]
      coPar.name=cParname
      oParlist[cParname]=coPar
    return oParlist
  def parse_function(self,iPar,iFunc):
    """ Parse the input function, insert parameters, return result """
    if iFunc['type']=='empirical':
      funcRes=muldata(iFunc['shape'],iPar['mul'].value)
    elif iFunc['type']=='analytical':
      funcRes=globals()[iFunc['shape']](self.target_x,iPar,self.psf)
    else:
      raise RuntimeError('Unknown function type!')
    return funcRes
  def extract_pars(self):
    """
    Extract the paramers from the function list so they can be manipulated by the residual minimization routines.
    """
    oPars=Parameters()
    for indFunc,cFunc in enumerate(self.funclist):
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