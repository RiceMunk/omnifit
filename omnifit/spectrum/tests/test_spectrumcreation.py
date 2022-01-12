# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
import numpy as np
from .. import spectrum
from ... import utils
from ...tests import helpers
from astropy import units as u


class TestSpectrumCreation_basic:
    def test_initbasic(self):
        """
        Make sure that basic base spectrum creation works as expected
        """
        testspec = helpers.generate_spectrum()
        assert np.any(testspec.x.value)
        assert np.any(testspec.y.value)

    def test_initnounit(self, recwarn):
        """
        Make sure that basic base spectrum creation works without units
        """
        xdata = np.arange(1000, 2000, 10)
        ydata = np.arange(0, 100, 1)
        testspec = spectrum.BaseSpectrum(xdata, ydata)
        w = recwarn.pop(RuntimeWarning)
        assert issubclass(w.category, RuntimeWarning)
        assert testspec.x.unit is not None
        assert testspec.y.unit is not None

    def test_initwithnondata(self):
        """
        Make sure that initialising with non-data specification goes through
        """
        xdata = np.arange(1000, 2000, 10)*u.micron
        ydata = np.arange(0, 100, 1)*utils.unit_od
        testspec = spectrum.BaseSpectrum(xdata, ydata, nondata=['dummy data'])
        assert 'dummy data' in testspec._BaseSpectrum__nondata

    def test_initwrongsize(self):
        """
        Make sure that initialising a spectrum with different sized x and y
        doesn't work
        """
        xdata = np.arange(1000, 2000, 1)*u.micron
        ydata = np.arange(0, 100, 1)*utils.unit_od
        with pytest.raises(RuntimeError):
            spectrum.BaseSpectrum(xdata, ydata)

    def test_initwithinf(self):
        """
        Make sure that infinities are converted to nans when creating spectrum
        """
        xdata = np.arange(1000, 2000, 10, dtype=float)*u.micron
        ydata = np.arange(0, 100, 1, dtype=float)*utils.unit_od
        ydata[3] = np.inf
        testspec = spectrum.BaseSpectrum(xdata, ydata)
        assert np.all(testspec.x == xdata)
        assert np.isnan(testspec.y[3])

    def test_initsort(self):
        """
        Make sure that unsorted data gets appropriately sorted.
        """
        xdata = np.arange(1000, 2000, 10, dtype=float)*u.micron
        ydata = np.arange(0, 100, 1, dtype=float)*utils.unit_od
        xdata_rev = xdata[::-1]  # reverse xdata
        testspec = spectrum.BaseSpectrum(xdata_rev, ydata)
        assert not np.all(testspec.x == xdata_rev)
        assert not np.all(testspec.y == ydata)
        assert np.all(testspec.x == xdata)
        assert np.all(testspec.y == ydata[::-1])


class TestSpectrumCreation_absorption:
    def test_initabs(self):
        """
        Make sure that absorption spectrum spectrum initialisation works as
        expected.
        """
        testspec = helpers.generate_absspectrum()
        assert testspec.y.value.any()

    def test_initwrongsize_abs(self):
        """
        As above, but deliberately mess up the size. This should raise an
        RuntimeError.
        """
        xdata = np.arange(1000, 2000, 10, dtype=float)*u.kayser
        ydata = np.arange(0, 100, 1, dtype=float)[1:]*utils.unit_od
        with pytest.raises(RuntimeError):
            spectrum.AbsorptionSpectrum(
                xdata,
                ydata,
                specname='test water spectrum (absorption)')

    def test_initlab(self):
        """
        Make sure that lab spectrum initialisation works as expected.
        """
        testspec = helpers.generate_cdespectrum()
        assert testspec.y.value.any

    def test_initwrongsize_lab(self):
        """
        As above, but deliberately mess up the size.
        This should raise an RuntimeError.
        """
        xdata = np.arange(1000, 2000, 10, dtype=float)*u.micron
        ndata = np.arange(0, 100, 1, dtype=float)[1:]
        kdata = np.arange(0, 100, 1, dtype=float)[1:]
        with pytest.raises(RuntimeError):
            spectrum.CDESpectrum(
                xdata[2:],
                np.vectorize(complex)(ndata, kdata),
                specname='test water spectrum (n and k)')
