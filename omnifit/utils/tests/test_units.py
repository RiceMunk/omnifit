# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
import numpy as np
import os
from ... import spectrum
from ... import utils
from ...tests import helpers
from astropy import units as u

class TestUnits:
  def test_unitconversions(self):
    
