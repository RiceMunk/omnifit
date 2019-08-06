# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
from ...tests.helpers import generate_spectrum
from ...tests.helpers import generate_cdespectrum
from ...tests.helpers import generate_absspectrum
from astropy import units as u


class TestSpectrumManipulation_conversion:
    def test_wl2wn2wl(self):
        """
        Make sure that conversion from microns to wavenumbers and back works
        """
        testspec = generate_spectrum()
        assert testspec.x.unit == u.micron
        testspec.convert2wn()
        assert testspec.x.unit == u.kayser
        testspec.convert2wl()
        assert testspec.x.unit == u.micron

    def test_customconvert(self):
        """
        Make sure that custom unit conversion works as expected
        """
        testspec = generate_spectrum()
        assert testspec.x.unit == u.micron
        testspec.convert2(u.meter)
        assert testspec.x.unit == u.meter
        with pytest.raises(u.UnitsError):
            testspec.convert2(u.kg)


class TestSpectrumManipulation_convolution:
    def test_gaussianconvolution(self):
        """
        Test the functionality of the gaussian convolution
        """
        testspec = generate_cdespectrum()
        oldunit_x = testspec.x.unit
        oldunit_y = testspec.y.unit
        testspec.gconvolve(10.)
        assert testspec.x.unit == oldunit_x
        assert testspec.y.unit == oldunit_y

    def test_repeatedconvolution(self, recwarn):
        """
        Make sure that a warning is raised if convolution is repeated
        """
        testspec = generate_cdespectrum()
        testspec.gconvolve(10.)
        testspec.gconvolve(10.)
        w = recwarn.pop(RuntimeWarning)
        assert issubclass(w.category, RuntimeWarning)

    def test_gaussianpsfistoolarge(self):
        """
        Make sure that gaussian PSF can't be made too large
        """
        testspec = generate_cdespectrum()
        with pytest.raises(Exception):
            testspec.gpsf(len(testspec.x)+1)

    def test_smooth(self):
        """
        Test the various smoothing convolutions
        """
        testspec = generate_cdespectrum()
        testspec.smooth(window='flat')
        testspec = generate_cdespectrum()
        testspec.smooth(window='hanning')
        testspec = generate_cdespectrum()
        testspec.smooth(window='hamming')
        testspec = generate_cdespectrum()
        testspec.smooth(window='bartlett')
        testspec = generate_cdespectrum()
        testspec.smooth(window='blackman')
        testspec = generate_cdespectrum()
        testspec.smooth(window_len=2)
        testspec = generate_cdespectrum()
        with pytest.raises(ValueError):
            testspec.smooth(window='not a window')
        testspec = generate_cdespectrum()
        with pytest.raises(ValueError):
            testspec.smooth(window_len=1e10)


class TestSpectrumManipulation_misc:
    def test_interpolate(self):
        """
        Test interpolation between two different spectra
        """
        # normal function
        testspec1 = generate_cdespectrum()
        testspec2 = generate_absspectrum()
        testspec1.interpolate(testspec2)
        # trying to break it
        testspec1 = generate_cdespectrum()
        testspec2 = generate_absspectrum()
        testspec2.x = testspec2.x.value * u.kg
        with pytest.raises(Exception):
            testspec1.interpolate(testspec2)
        testspec1 = generate_cdespectrum()
        testspec2 = generate_absspectrum()
        testspec2.y = testspec2.y.value * u.kg
        with pytest.raises(Exception):
            testspec1.interpolate(testspec2)
        # test that units are retained on interpolation
        testspec1 = generate_cdespectrum()
        testspec2 = generate_absspectrum()
        oldunit_y = testspec1.y.unit
        oldunit_x = testspec1.x.unit
        testspec1.interpolate(testspec2)
        assert testspec1.x.unit == oldunit_x
        assert testspec1.y.unit == oldunit_y

    def test_subspectrum(self):
        """
        Test the extraction of a subspectrum from a spectrum
        """
        testspec = generate_cdespectrum()
        testspec.subspectrum(
            testspec.x[0].value+500,
            testspec.x[-1].value-500)

    def test_baseline_basic(self):
        """
        Test the baselining of a spectrum
        """
        testspec = generate_cdespectrum()
        testspec.baseline()
        testspec = generate_cdespectrum()
        testspec.baseline(
            windows=[[testspec.x[0].value+500, testspec.x[-1].value-500]],
            exclusive=True)
