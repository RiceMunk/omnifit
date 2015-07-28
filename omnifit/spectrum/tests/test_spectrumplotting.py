# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
import numpy as np
import os
from ..spectrum import *
import matplotlib.pyplot as plt
from ...tests.helpers import *

class TestSpectrumPlotting:
  def test_plotbasic(self):
    """
    Make sure that basic spectrum plotting works as expected
    """
    testspec = generate_spectrum()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    testspec.plot(ax)
    testspec.plot(ax,drawstyle='steps-post')
    plt.close()
  def test_plotwrong(self):
    """
    Make sure that plotting fails when it should
    """
    testspec = generate_spectrum()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(Exception):
      testspec.plot(ax,plotstyle='non-existent style')
    with pytest.raises(Exception):
      testspec.plot(ax,x='baselined')
  def test_plotnk(self):
    """
    Make sure that n and k spectrum plotting works as expected
    """
    testspec = generate_labspectrum()
    #no range
    fig = testspec.plotnk()
    assert not(fig is None)
    plt.close()
    #with range
    fig = testspec.plotnk(xrange=[3000,3050])
    assert not(fig is None)
    plt.close()
  def test_plotabs(self):
    """
    Make sure that OD spectrum plotting works as expected
    """
    testspec = generate_absspectrum()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    testspec.plotod(ax,in_wl=False)
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    testspec.plotod(ax,in_wl=True)
    plt.close()
