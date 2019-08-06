# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from ... import utils
from astropy import units as u


class TestUnits:
    def test_unitconversions(self):
        transmittance = 10.0
        absorbance = -np.log10(transmittance)
        opticaldepth = absorbance*np.log(10)
        transmittance *= utils.unit_transmittance
        absorbance *= utils.unit_absorbance
        opticaldepth *= utils.unit_opticaldepth
        with u.set_enabled_equivalencies(utils.equivalencies_absorption):
            assert absorbance.to(utils.unit_transmittance).value == \
                transmittance.value
            assert absorbance.to(utils.unit_opticaldepth).value == \
                opticaldepth.value

            assert transmittance.to(utils.unit_absorbance).value == \
                absorbance.value
            assert transmittance.to(utils.unit_opticaldepth).value == \
                opticaldepth.value

            assert opticaldepth.to(utils.unit_absorbance).value == \
                absorbance.value
            assert opticaldepth.to(utils.unit_transmittance).value == \
                transmittance.value
