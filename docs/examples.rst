Examples
========
This page lists examples of likely ways to make use of Omnifit.

Fitting from start to finish
----------------------------
The code snippet below shows how one might go about fitting a target spectrum with an empirically acquired spectrum and a Gaussian function, and the both export the raw fit results and plot them.

The target spectrum data is assumed to originate from a file which consists of two columns of data. The first column contains the wavelength and the second column contains the optical depth of an observed spectrum.

The empirical spectrum being fitted to this, by contrast, is assumed to be a file containing set of data describing the complex refractive index of an ice. The first column of this file contains the frequency of the spectrum (in reciprocal wavenumbers), and the second and third columns containing the real and imaginary parts of the complex refractive index.

.. code-block:: python
  :linenos:

  import numpy as np
  from omnifit.spectrum import AbsorptionSpectrum,CDEspectrum
  from omnifit.utils import unit_od
  import astropy.units as u
  from omnifit.fitter import Fitter
  from lmfit import Parameters
  obs_wl,obs_od = np.loadtxt('./obsdata.dat',dtype=float,usecols=(0,1))
  lab_wn,lab_n,lab_k = np.loadtxt('./labdata.dat',dtype=float,usecols=(0,1,2))
  obs_spec = AbsorptionSpectrum(obs_wl*u.micron,obs_od*unit_od,specname='Observed data')
  lab_spec = CDESpectrum(lab_wn,lab_n,lab_k,specname='Laboratory data')
  interp_lab = lab_spec.interpolate(obs_spec)
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

In this, line 1-6 import the various components needed for the full fit.
Lines 7 and 8 read the spectrum data from the two files which contain the target spectrum and the complex refractive index spectrum.

Line 9 initialises the target spectrum as an absorption spectrum using the data read in line 7, and line 10 initialises the CDE-corrected spectrum using the data read in line 8.

Line 11 interpolates the data in the CDE-corrected spectrum to match the spectral resolution of the fitting target spectrum, which is also used to initialise the fitter on line 12.

Lines 13 and 14 prepare the fitting parameters for an empirical data, containing only an initial guess (0.5) for the best-fit multiplier. These parameters and the data to be fitted are then given to the fitter on line 15.

Similarly lines 16-19 involve setting up the initial guess and constrains on the parameters for a Gaussian fit, which are then given to the fitter (with instruction to try and fit a Gaussian) on line 20.

Line 21 is used to make the fitting itself happen. After it is done, the command on line 22 can be called to produce two files, example_fitres.csv and example_fitres.xml, which contain all the information on the fit results, as documented in `omnifit.fitter.Fitter.fitresults_tofile`.

Finally, lines 23-26 are used to produce a plot of the fit results in the file example_fitres.pdf.