import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.integrate import simps
from sys import float_info
import warnings



class Baseliner:
  """
  A class for interactive baseliner of spectroscopic data.

  The class works by being fed a spectrum and a matplotlib axis on which
  it should be plotted. The spectrum is then plotted to the given axis,
  and a number of interactive options are made available to the user.

  Left-clicking with the mouse for the first time starts defining a window
  from the x-axis location of the click. A second click finishes the
  window between the locations of the first and second click.
  A third click will finish selecting windows, and perform the baselining.
  Alternately, right-clicking will cancel the last left-clicking action,
  allowing misplaced windows to be adjusted.

  Two keys are also accepted:
  Pressing "q" will cause the baselining process to be canceled,
  effectively skipping the baselining of this spectrum.
  Pressing "a" will allow an additional window to be defined, assuming
  one has been defined so far (by left-clicking twice to define its 
  boundaries).

  Attributes
  ----------
  windows : `list`
    A list of all the set windows.
  """
  def __init__(self,ax,spec):
    """
    Baseliner(ax,spec,order=1)

    Initialise the `Baseliner` class by giving it the target axis and
    spectrum.

    Parameters
    ----------
    ax : `matplotlib.axis`
      The matplotlib axis on which the interation will happen.
    spec : `omnifit.spectrum.BaseSpectrum`
      The spectrum which will be plotted as the visual reference on
      the given axis.
    """
    self.__ax = ax
    self.__spec = spec
    self.__x = spec.x.value
    self.__y = spec.y.value
    self.__limlo=None
    self.__limhi=None
    self.__minx=np.min(self.__x)
    self.__maxx=np.max(self.__x)
    self.__miny=np.min(self.__y)
    self.__maxy=np.max(self.__y)
    self.__ax.set_xlim(self.__minx,self.__maxx)
    self.__ax.set_ylim(self.__miny,self.__maxy)
    self.__specplot,=self.__ax.plot(self.__x,self.__y,'k-',drawstyle='steps-mid')
    self.__buttonListener = self.__ax.figure.canvas.mpl_connect('button_press_event', self.__mouse_press)
    self.__keyListener = self.__ax.figure.canvas.mpl_connect('key_press_event', self.__key_press)
    self.windows=[]
  def __key_press(self, event):
    if event.key=='q':
      self.__skip()
    if event.key=='a' and self.__limlo != None and self.__limhi != None:
      self.__addwindow(self.__limlo,self.__limhi)
      self.__ax.plot([self.__limlo,self.__limlo],[self.__miny,self.__maxy],'g-')
      self.__ax.plot([self.__limhi,self.__limhi],[self.__miny,self.__maxy],'g-')
      self.__remlim()
      self.__remlim()
      print 'Window added. Ready to receive another one.'
    else:
      return
  def __mouse_press(self, event):
    if event.button==1:
      self.__setlim(event.xdata)
    elif event.button==2:
      return
    elif event.button==3:
      self.__remlim()
  def __skip(self):
    plt.close()
  def __setlim(self,i_x):
    if self.__limlo==None:
      self.__limlo=i_x
      self.__limloplot,=self.__ax.plot([i_x,i_x],[self.__miny,self.__maxy],'b-')
      self.__ax.figure.canvas.draw()
    elif self.__limhi==None:
      self.__limhi=i_x
      self.__limhiplot,=self.__ax.plot([i_x,i_x],[self.__miny,self.__maxy],'b-')
      self.__ax.figure.canvas.draw()
      print 'Ready for finalising. Press once more to do so, or press a to add another window.'
    else:
      self.__finalise()
  def __remlim(self):
    if self.__limhi!=None:
      self.__limhi=None
      self.__limhiplot.set_ydata([self.__miny,self.__miny])
      self.__ax.figure.canvas.draw()
    elif self.__limlo!=None:
      self.__limlo=None
      self.__limloplot.set_ydata([self.__miny,self.__miny])
      self.__ax.figure.canvas.draw()
    else:
      print 'No limits to cancel.'
  def __addwindow(self,limlo,limhi):
    if limhi < limlo:
      limlo,limhi = limhi,limlo
    self.windows.append([limlo,limhi])
  def __finalise(self):
    self.__addwindow(self.__limlo,self.__limhi)
    self.__ax.figure.canvas.mpl_disconnect(self.__buttonListener)
    self.__ax.figure.canvas.mpl_disconnect(self.__keyListener)
    plt.close(self.__ax.figure)

#---------------------
#New units definitions
#---------------------
#the units themselves
unit_t = u.def_unit('transmittance units',doc='Transmittance of radiation')
unit_transmittance = unit_t
unit_abs = u.def_unit('absorbance units',doc='Absorbance of radiation')
unit_absorbance = unit_abs
unit_od = u.def_unit('optical depth units',doc='Optical depth of radiation')
unit_opticaldepth = unit_od

#the equivalencies between the units
equivalencies_absorption = [
    (unit_t,unit_abs,lambda x:-np.log10(x),lambda x:10**-x),
    (unit_od,unit_abs,lambda x:x/np.log(10),lambda x:x*np.log(10)),
    (unit_od,unit_t,lambda x:10**(-x/np.log(10)),lambda x:-np.log10(x)*np.log(10))
    ]

#------------------------------------------------------
#Functions related to light scattering and transmission
#------------------------------------------------------
def cde_correct(wn,n,k):
  """
  cde_correct(wn,n,k)

  Generate a CDE-corrected spectrum from a complex refractive index
  spectrum.

  Parameters
  ----------
  wn : `numpy.ndarray`
    The frequency data of the input spectrum, in reciprocal
    wavenumbers (cm^-1).
  n : `numpy.ndarray`
    The real component of the complex refractive index spectrum.
  k : `numpy.ndarray`
    The imaginary component of the complex refractive index spectrum.

  Returns
  -------
  A list containing the following numpy arrays, in given order:
    * The spectrum of the absorption cross section of the simulated grain.
    * The spectrum of the absorption cross section of the simulated grain,
      normalized by the volume distribution of the grain. This parameter
      is the equivalent of optical depth in most cases.
    * The spectrum of the scattering cross section of the simulated grain,
      normalized by the volume distribution of the grain.
    * The spectrum of the total cross section of the simulated grain.    
  """
  wl=1.e4/wn
  m=np.vectorize(complex)(n,k)
  m2=m**2.0
  im_part=((m2/(m2-1.0))*np.log(m2)).imag
  cabs_vol=(4.0*np.pi/wl)*im_part
  cabs=wn*(2.0*m.imag/(m.imag-1))*np.log10(m.imag)
  cscat_vol=(wn**3.0/(6.0*np.pi))*cabs
  ctot=cabs+cscat_vol
  return cabs,cabs_vol,cscat_vol,ctot

def complex_transmission_reflection(in_m0,in_m1,in_m2):
  """
  Calculate the complex transmission and reflection coefficients between
  media 0, 1, and 2 given their complex refractive indices.
  In the Kramers-Kronig implementation media 0, 1, and 2 correspond
  respectively to the vacuum, ice, and substrate.
  """
  complex_transmission = lambda m1,m2: (2.*m1.real)/(m1+m2)
  complex_reflection = lambda m1,m2: (m1-m2)/(m1+m2)
  return (
          complex_transmission(in_m0,in_m1),
          complex_transmission(in_m0,in_m2),
          complex_transmission(in_m1,in_m2),
          complex_reflection(in_m0,in_m1),
          complex_reflection(in_m0,in_m2),
          complex_reflection(in_m1,in_m2)
        )

def kramers_kronig(freq,transmittance,m_substrate,d_ice,m0,freq_m0,m_guess=1.0+0.0j,tol=0.001,maxiter=100,ignore_fraction=0.1,force_kkint_unity=False,precalc=False):
  """
  kramers_kronig()
  Kramers-Kronig relation.
  This is an implementation of the Kramers-Kronig relation calculation
  presented in Hudgins et al 1993 (1993ApJS...86..713H).
  Parameters
  ----------
  Returns
  -------
  """
  #set up constants
  m_vacuum = 1.0+0.0j
  #make sure the input array units are correct; convert if necessary
  if type(freq) != u.quantity.Quantity:
    warnings.warn('No units detected in input freq. Assuming kayser.',RuntimeWarning)
    freq *= u.kayser
  else:
    with u.set_enabled_equivalencies(u.equivalencies.spectral()):
      freq=freq.to(u.kayser)
  if type(transmittance) != u.quantity.Quantity:
    warnings.warn('No units detected in input transmittance. Assuming transmittance units.',RuntimeWarning)
    transmittance *= unit_t
  else:
    with u.set_enabled_equivalencies(equivalencies_absorption):
      transmittance = transmittance.to(unit_t)
  if type(d_ice) != u.quantity.Quantity:
    warnings.warn('No units detected in input d_ice. Assuming centimeters.',RuntimeWarning)
    d_ice *= u.cm
  else:
    d_ice = d_ice.to(u.cm)
  #sort the arrays and get rid of units; won't need them after this
  initial_sorter = np.argsort(freq)
  freq = freq[initial_sorter].value
  transmittance = transmittance[initial_sorter].value
  d_ice = d_ice.value
  #initialise complex refractive index and alpha arrays
  m = np.full_like(freq,np.nan+np.nan*1j,dtype=complex)
  alpha = np.full_like(freq,np.nan+np.nan*1j,dtype=complex)
  #initial guess at m at first index
  m_ice = np.full_like(freq,m_guess,dtype=complex)
  #find top and bottom fraction indices. These will be replaced with dummy values after each integration to get rid of edge effects
  if ignore_fraction > 0.5 or ignore_fraction < 0:
    raise RuntimeError('ignore_fraction must be between 0.0 and 0.5')
  bot_fraction = round(ignore_fraction*len(freq))
  top_fraction = len(freq)-bot_fraction
  #pre-calculate the large denominator component of the KK integration, if desired
  if precalc:
    try:
      sfreq=(freq).reshape(len(freq),1)
      kkint_deno1 = freq**2-sfreq**2
      kkint_deno1[kkint_deno1!=0] = 1./kkint_deno1[kkint_deno1!=0]
      precalc = True
    #or at least try to do so; if run out of memory, switch to the slower no-precalc mode
    except MemoryError:
      precalc = False
  #some other parts can always be precalced
  kkint_mul = 1./(2*np.pi*np.pi)
  kkint_deno2 = freq**2-freq_m0**2
  kkint_deno2[kkint_deno2!=0] = 1./kkint_deno2[kkint_deno2!=0]
  #calculate alpha at freq0
  alpha0 = m0.imag/(4*np.pi*freq)
  #iteration begin!
  niter = 0
  squaresum_diff = tol+1
  while squaresum_diff > tol and niter < maxiter:
    #calculate transmission and relfection coefficients
    #in these 0 means vacuum, 1 means ice, 2 means substrate
    t01,t02,t12,r01,r02,r12 = complex_transmission_reflection(m_vacuum,m_ice,m_substrate)
    #the reflection component
    # reflection_component = np.abs((t01*t12/t02)/(1.+r01*r12*np.exp(4.j*np.pi*d_ice*m_ice*freq)))**2.)
    #this is an evil equation. do NOT touch it
    #it calculates the lambert absorption coefficient using the current best guess at m_ice
    alpha = (1./d_ice)*(-np.log(transmittance)+np.log(np.abs((t01*t12/t02)/(1.+r01*r12*np.exp(4.j*np.pi*d_ice*m_ice*freq)))**2.))
    #using the new alpha, calculate a new n (and thus m) for the ice
    #this is done in a parallel for loop, to avoid killing the computer when dealing with large amounts of data
    kkint_nomi = alpha-alpha0
    kkint = np.full_like(alpha,m0.real)
    numcols = kkint_nomi.shape[0]
    for current_col in range(numcols):
      if precalc:
        kkint[current_col]+=kkint_mul*simps((alpha-alpha[current_col])*kkint_deno1[current_col,:]-kkint_nomi*kkint_deno2)
      else:
        kkint_deno1 = freq[current_col]**2-freq**2
        kkint_deno1[kkint_deno1!=0] = 1./kkint_deno1[kkint_deno1!=0]
        kkint[current_col]+=kkint_mul*simps((alpha-alpha[current_col])*kkint_deno1-kkint_nomi/(freq**2-freq_m0**2))
    if np.any(kkint<1):
      if np.any(kkint<0):
        warnings.warn('KK integration is producing negative refractive indices! This will most likely produce nonsensical results.',RuntimeWarning)
      else:
        warnings.warn('KK integration is producing refractive indices below unity! This may result in unexpected behaviour.',RuntimeWarning)
      if force_kkint_unity:
        kkint[kkint<1]=1.
    m_ice = kkint+1j*alpha/(4*np.pi*freq)
    if np.any(np.isnan(m_ice.real)) or np.any(np.isnan(m_ice.imag)):
      raise RuntimeError('Produced complex refractive index contains NaNs. Check your input parameters.')
    #replace top and bottom fractions of m_ice with the value closest to that edge
    #this is done to combat edge effects arising from integrating over a non-infinite range
    m_ice[:bot_fraction] = m_ice[bot_fraction]
    m_ice[top_fraction:] = m_ice[top_fraction]
    #calculate transmission and relfection coefficients (again)
    #in these 0 means vacuum, 1 means ice, 2 means substrate
    t01,t02,t12,r01,r02,r12 = complex_transmission_reflection(m_vacuum,m_ice,m_substrate)
    #model a transmittance using given m_ice and alpha
    #yes, this is another evil equation
    transmittance_model = np.exp(-alpha*d_ice)*np.abs((t01*t12/t02)/(1.+r01*r12*np.exp(4.j*np.pi*d_ice*m_ice*freq)))**2.
    diff = transmittance - transmittance_model
    diff[:bot_fraction] = 0. #ignore top...
    diff[top_fraction:] = 0. #...and bottom fraction differences
    squaresum_diff = np.sum(diff**2) #square sum of difference
    niter += 1
  #at this point we are done
  if niter>=maxiter:
    raise RuntimeError('Maximum number of iterations reached before convergence criterion was met.')
  return m_ice