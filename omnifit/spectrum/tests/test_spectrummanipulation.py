# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
import numpy as np
import os
from ..spectrum import *
from ...tests.helpers import *

class TestSpectrumManipulation_conversion:
  def test_wl2wn2wl(self):
    """
    Make sure that conversion from microns to wavenumbers and back works
    """
    testspec = generate_spectrum()
    original_x = testspec.x
    testspec.xUnit = 'Wavelength'
    testspec.wl2wn()
    testspec.wn2wl()
    assert np.all(np.abs(testspec.x - original_x) < epsilon)
  def test_wl2wn2wl_fail(self):
    testspec = generate_spectrum()
    with pytest.raises(Exception):
      testspec.wn2wl()
    with pytest.raises(Exception):
      testspec.wl2wn()

class TestSpectrumManipulation_convolution:
  def test_gaussianconvolution(self):
    """
    Test the functionality of the gaussian convolution
    """
    testspec = generate_labspectrum()
    testspec.gconvolve(10.)
  def test_repeatedconvolution(self,recwarn):
    """
    Make sure that a warning is raised if convolution is repeated
    """
    testspec = generate_labspectrum()
    testspec.gconvolve(10.)
    testspec.gconvolve(10.)
    w = recwarn.pop(RuntimeWarning)
    assert issubclass(w.category, RuntimeWarning)
  def test_gaussianpsfistoolarge(self):
    """
    Make sure that gaussian PSF can't be made too large
    """
    testspec = generate_labspectrum()
    with pytest.raises(Exception):
      testspec.gpsf(len(testspec.x)+1)
  def test_autopsf(self):
    """
    Test the functionality of autopsf
    """
    testspec = generate_labspectrum()
    testspec.autopsf()
  def test_smooth(self):
    """
    Test the various smoothing convolutions
    """
    testspec = generate_labspectrum()
    testspec.smooth(window='flat')
    testspec = generate_labspectrum()
    testspec.smooth(window='hanning')
    testspec = generate_labspectrum()
    testspec.smooth(window='hamming')
    testspec = generate_labspectrum()
    testspec.smooth(window='bartlett')
    testspec = generate_labspectrum()
    testspec.smooth(window='blackman')
    testspec = generate_labspectrum()
    testspec.smooth(window_len=2)
    testspec = generate_labspectrum()
    with pytest.raises(ValueError):
      testspec.smooth(window='not a window')
    testspec = generate_labspectrum()
    with pytest.raises(ValueError):
      testspec.smooth(window_len=1e10)

class TestSpectrumManipulation_misc:
  def test_interpolate(self):
    """
    Test interpolation between two different spectra
    """
    #normal function
    testspec1 = generate_labspectrum()
    testspec2 = generate_absspectrum()
    testspec1.interpolate(testspec2)
    #trying to break it
    testspec1 = generate_labspectrum()
    testspec2 = generate_absspectrum()
    testspec2.xUnit = 'fake x units'
    with pytest.raises(Exception):
      testspec1.interpolate(testspec2)
    testspec1 = generate_labspectrum()
    testspec2 = generate_absspectrum()
    testspec2.yUnit = 'fake y units'
    with pytest.raises(Exception):
      testspec1.interpolate(testspec2)
  def test_subspectrum(self):
    """
    Test the extraction of a subspectrum from a spectrum
    """
    testspec = generate_labspectrum()
    testspec.subspectrum(testspec.x[0]+500,testspec.x[-1]-500)
  def test_baseline_basic(self):
    """
    Test the baselining of a spectrum
    """
    testspec = generate_labspectrum()
    testspec.baseline()
    testspec = generate_labspectrum()
    testspec.baseline(windows=[[testspec.x[0]+500,testspec.x[-1]-500]],exclusive=True)
  def test_baseline_manual(self):
    """
    Test manual baseline functionality
    as far as the interactive mode allows non-interactively
    """
    testspec = generate_labspectrum()
    cFig,cBaseliner = testspec.baseline(windows='manual')
    from matplotlib.backend_bases import MouseEvent, KeyEvent
    leftclick = MouseEvent('button_press_event',cFig.canvas,2000.,0.,button=1)
    midclick = MouseEvent('button_press_event',cFig.canvas,1000.,0.,button=2)
    rightclick = MouseEvent('button_press_event',cFig.canvas,1000.,0.,button=3)
    keypress_a = KeyEvent('key_press_event',cFig.canvas,'a')
    keypress_q = KeyEvent('key_press_event',cFig.canvas,'q')
    keypress_other = KeyEvent('key_press_event',cFig.canvas,'@')
    cBaseliner.mouse_press(leftclick)
    cBaseliner.mouse_press(midclick)
    cBaseliner.mouse_press(rightclick)
    cBaseliner.setlim(2000.)
    cBaseliner.setlim(2500.)
    cBaseliner.key_press(keypress_a)
    cBaseliner.key_press(keypress_other)
    cBaseliner.setlim(2800.)
    cBaseliner.setlim(2600.)
    cBaseliner.setlim(0.)
    # windows = [[0.0,1.0e6]]
