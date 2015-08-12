# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
import numpy as np
import os
from ..spectrum import *
from ...tests.helpers import *

class TestSpectrumInfo:
  def test_minmax(self):
    """
    Make sure that min and max work
    """
    testspec = generate_cdespectrum()
    testspec.min()
    testspec.min(checkrange=[2000.,2050.])
    testspec.max()
    testspec.max(checkrange=[2000.,2050.])
  def test_info(self):
    """
    Make sure that spectrum info is produced
    """
    testspec = generate_cdespectrum()
    testspec.info()