# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
import numpy as np
import os
from ..spectrum import *
from ...tests.helpers import *

class TestSpectrumCreation_basic:
  def test_initbasic(self):
    """
    Make sure that basic base spectrum creation works as expected
    """
    testspec = generate_spectrum()
    assert np.any(testspec.x)
    assert np.any(testspec.y)
  def test_initwithnondata(self):
    """
    Make sure that initialising with non-data specification goes through
    """
    xdata = np.arange(1000,2000,10)
    ydata = np.arange(0,100,1)
    testspec = spectrum(xdata,ydata,nonData=['dummy data'])
    assert 'dummy data' in testspec.nonData
  def test_initwrongsize(self):
    """
    Make sure that initialising a spectrum with different sized x and y doesn't work
    """
    xdata = np.arange(1000,2000,1)
    ydata = np.arange(0,100,1)
    with pytest.raises(Exception):
      testspec = spectrum(xdata,ydata)
  def test_initwithnan(self):
    """
    Make sure that nans are dropped when creating spectrum
    """
    xdata = np.arange(1000,2000,10,dtype=np.float)
    ydata = np.arange(0,100,1,dtype=np.float)
    xdata[3] = np.nan
    testspec = spectrum(xdata,ydata)
    assert not np.all(testspec.x == xdata)
    assert not np.all(testspec.y == ydata)
    notnan_indices = [i for i,j in enumerate(xdata) if i!=3]
    assert np.all(testspec.x == xdata[notnan_indices])
    assert np.all(testspec.y == ydata[notnan_indices])
  def test_initsort(self):
    """
    Make sure that unsorted data gets appropriately sorted.
    """
    xdata = np.arange(1000,2000,10,dtype=np.float)
    ydata = np.arange(0,100,1,dtype=np.float)
    xdata_rev = xdata[::-1] #reverse xdata
    testspec = spectrum(xdata_rev,ydata)
    assert not np.all(testspec.x == xdata_rev)
    assert not np.all(testspec.y == ydata)
    assert np.all(testspec.x == xdata)
    assert np.all(testspec.y == ydata[::-1])

class TestSpectrumCreation_absorption:
  def test_initabs(self):
    """
    Make sure that absorption spectrum spectrum initialisation works as expected.
    """
    testspec = generate_absspectrum()
    assert np.any(testspec.y)
  def test_initwrongsize_abs(self):
    """
    As above, but deliberately mess up the size. This should raise an exception.
    """
    xdata = np.arange(1000,2000,10,dtype=np.float)
    ydata = np.arange(0,100,1,dtype=np.float)[1:]
    with pytest.raises(Exception):
      testspec = absorptionSpectrum(xdata,ydata,specname='test water spectrum (absorption)')
  def test_initlab(self):
    """
    Make sure that lab spectrum initialisation works as expected.
    """
    testspec = generate_labspectrum()
    assert np.any(testspec.y)
  def test_initwrongsize_lab(self):
    """
    As above, but deliberately mess up the size. This should raise an exception.
    """
    xdata = np.arange(1000,2000,10,dtype=np.float)
    ndata = np.arange(0,100,1,dtype=np.float)[1:]
    kdata = np.arange(0,100,1,dtype=np.float)[2:]
    with pytest.raises(Exception):
      testspec = labSpectrum(xdata,ndata,kdata,specname='test water spectrum (n and k)')