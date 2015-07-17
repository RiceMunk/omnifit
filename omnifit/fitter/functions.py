import numpy as np

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