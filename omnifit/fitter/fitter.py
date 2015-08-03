import numpy as np
from lmfit import minimize, Parameters, Parameter
from astropy import units as u
from .. import spectrum
from functions import *

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
      raise RuntimeError('Input arrays have different sizes.')
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
      raise RuntimeError('Input spectrum has wrong size!')
    self.funcList.append({'type':'lab','shape':iSpectrum.y,'params':iParams,'name':funcname,'color':color})
  def add_theory(self,iShape,iParams,funcname='Unknown function',color='red'):
    """
    Add theoretical function to fitting pool.
    Uses lmfit params as the parameter list
    """
    self.funcList.append({'type':'theory','shape':iShape,'params':iParams,'name':funcname,'color':color})

  def perform_fit(self):
    """
    Perform least-squares fitting to the function list
    """
    self.fitPars = self.extract_pars()
    self.fitRes=minimize(self.fit_residual,self.fitPars,epsfcn=0.05)
    if not(self.fitRes.success):
      raise RuntimeError('Fitting failed!')
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
      funcRes=muldata(iFunc['shape'],iPar['mul'].value)
    elif iFunc['type']=='theory':
      funcRes=globals()[iFunc['shape']](self.targX,iPar,self.psf)
    else:
      raise RuntimeError('Unknown function type!')
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
