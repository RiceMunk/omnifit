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
    testspec.plot(ax,drawstyle='steps-mid')
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
    testspec = generate_cdespectrum()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig = testspec.plotnk(ax1,ax2)
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
