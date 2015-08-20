Examples
========
This page lists examples of likely ways to make use of Omnifit, and 

CDE correction
--------------

Fitting from start to finish
----------------------------
The code snippet below shows how one might go about fitting a target spectrum with an empirically acquired spectrum and a Gaussian function, and the both export the raw fit results and plot them.

The target spectrum data is assumed to originate from a file which consists of two columns of data. The first column contains the wavelength and the second column contains the optical depth of an observed spectrum.

The empirical spectrum being fitted to this, by contrast, is assumed to be a file containing set of data describing the complex refractive index of an ice. The first column of this file contains the frequency of the spectrum (in reciprocal wavenumbers), and the second and third columns containing the real and imaginary parts of the complex refractive index.

.. code-block:: python
  :linenos:

  import numpy as np
  obs_wl,obs_od = np.loadtxt('./obsdata.dat',dtype=float,usecols=(0,1))
  lab_wn,lab_n,lab_k = np.loadtxt('./labdata.dat',dtype=float,usecols=(0,1,2))
  from omnifit.spectrum import AbsorptionSpectrum,CDEspectrum
  from omnifit.utils import unit_od
  import astropy.units as u
  obs_spec = AbsorptionSpectrum(obs_wl*u.micron,obs_od*unit_od,specname='Observed data')
  lab_spec = CDESpectrum(lab_wn,lab_n,lab_k,specname='Laboratory data')
  interp_lab = lab_spec.interpolate(obs_spec)
  from omnifit.fitter import Fitter
  from lmfit import Parameters
  fitter_example = Fitter.fromspectrum(obs_spec,modelname='Example fit')
  lab_par = Parameters()
  lab_par.add('mul',value=0.5,min=0.0)
  fitter_example.add_empirical(interp_lab,lab_par,funcname='Example empirical function')
  theory_par=Parameters()
  theory_par.add_many(('peak',  2.5,   True, 0.0,        None,       None),
                      ('pos',   3000., True, 3000.-200., 3000.+200., None),
                      ('fwhm',  50.0,  True, 0.0,        None,       None))
  fitter_example.add_analytical('gaussian',theory_par,funcname='Example gaussian')
  fitter_example.perform_fit()
  fitter_example.fitresults_tofile('example_fitres')
  import matplotlib.pyplot as plt
  ax = plt.subplots()
  fitter_example.plot_fitresults(ax)
  plt.savefig('example_fitres.pdf')

In this, line 1-3 imports omnifit, numpy and matplotlib. Strictly speaking, importing numpy and matplotlib is not necessary because Omnifit already imports them, but this way we prevent any ambiguity when calling either package in this example session. Lines 4 and 5 involve reading the observational and laboratory datas into their respective arrays. On the observational data file, columns 1 and 2 contain the wavelength and optical depth, respectively. For the laboratory data file columns 1-3 contain the frequency (in cm^-1), and n and k values respectively. Line 6 initializes the observational spectrum in the spectrum class, while line 7 calculates a CDE-corrected spectrum of the laboratory data. Line 8 creates a version of the laboratory spectrum which is interpolated to match the spectral resolution of the observation spectrum. Line 9 initializes the fitter class with the x (wavenumber) and y (optical depth) data of the observational spectrum. Lines 10-13 add the interpolated laboratory spectrum to the collection of fittable functions with an initial guess of the multiplier at 0.5. Lines 14-20 add a Gaussian to the function collection, with initial guesses of 2.5, 3000.0, and 50.0 for the peak, centroid position and full width half-maximum of the function, and with a constraint on the centroid position which permits it to deviate only up to 200 cm^-1 from the initial guess. Line 21 performs the actual fit, and line 22 saves the fit results to two files starting with "example_fitres". The files created are "example_fitres.xml" which contains various information (such as the best-fit parameters) and "example_fitres.csv" which contains the x,y information of the target data and the best-fitted data each separated to their own columns. Finally, lines 23-27 plot and save the fit results to the file "example_fitres.pdf".