# Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib.pyplot as plt
from ...tests.helpers import generate_cdespectrum
from ..fitter import Fitter
from lmfit import Parameters
from astropy import convolution


class TestFitterFitting:
    def test_fitterinit(self):
        """
        Make sure that fitter initialises as it should
        """
        testspec = generate_cdespectrum()
        Fitter(testspec.x.value, testspec.y.value)

    def test_fitterinit_fromspectrum(self):
        """
        Make sure that fitter initialises as it should when
        initialising with a spectrum
        """
        testspec = generate_cdespectrum()
        Fitter.fromspectrum(testspec)

    def test_fitlab(self):
        """
        Test the fitting of lab data
        """
        testspec = generate_cdespectrum()
        testfitter = Fitter(testspec.x.value, 2.*testspec.y.value)
        testpars = Parameters()
        #                 (Name,  Value,  Vary,   Min,     Max,     Expr)
        testpars.add_many(('mul', 1.0,    True,   0.0,     None,    None))
        testfitter.add_empirical(testspec, testpars)
        testfitter.perform_fit()

    def test_fittheory(self):
        """
        Test the fitting of the available analytical functions
        """
        testspec = generate_cdespectrum()
        testspec = testspec.subspectrum(2500., 3700., clone=True)
        testfitter = Fitter.fromspectrum(testspec)
        testpars = Parameters()
        #        (Name,    Value,  Vary,   Min,    Max,     Expr)
        testpars.add_many(
                 ('H',   0.1,   True, 0.0,       None, None, None),
                 ('xR',  3000., True, 3000.-50., 3000.+50., None, None),
                 ('w',   50.0,  True, 0.0,       None, None, None),
                 ('tau', 50.0,  True, 0.0,       None, None, None)
                 )
        testfitter.add_analytical(
            'flipped_egh',
            testpars, funcname='test fEGH')
        testpars = Parameters()
        #    (Name,    Value,  Vary,   Min,    Max,     Expr)
        testpars.add_many(
             ('peak',   1.0,     True,   0.0,        None, None, None),
             ('pos',    3300., True,   3300.-300.,  3300.+300., None, None),
             ('fwhm',   500.0,  True,    0.0,        None, None, None),
             )
        testfitter.add_analytical(
            'gaussian',
            testpars,
            funcname='test gaussian')
        testfitter.perform_fit()

    def test_fittheory_convolved(self):
        """
        Test the fitting of the available analytical
        functions with convolution enabled
        """
        testspec = generate_cdespectrum()
        testspec1 = testspec.subspectrum(2000., 2300., clone=True)
        testpsf1 = convolution.Gaussian1DKernel(5)
        testfitter1 = Fitter.fromspectrum(testspec1, psf=testpsf1)
        testpars = Parameters()
        #               (Name,    Value,  Vary,  Min,    Max, Expr)
        testpars.add_many(
                        ('lor1',   1.67, False,  None,   None,   None),
                        ('lor2',   195., False,   None,   None,   None),
                        ('lor3',    1.5, False,   None,   None,   None),
                        ('peak', 0.05,  True,   0.0,    0.1,    None),
                        ('pos',  2139.9,  True,   2129.9, 2149.9, None))
        testfitter1.add_analytical(
            'cde_lorentzian',
            testpars,
            funcname='test lorentzian')
        testfitter1.perform_fit()

        testspec2 = testspec.subspectrum(2500., 3700., clone=True)
        testpsf2 = convolution.Gaussian1DKernel(5)
        testfitter2 = Fitter.fromspectrum(testspec2, psf=testpsf2)
        testpars = Parameters()
        #            (Name,    Value,  Vary,   Min,    Max,     Expr)
        testpars.add_many(
                     ('H',      1.0,    True,   0.0,        None, None),
                     ('xR',     3000.,  True,   3000.-50.,  3000.+50., None),
                     ('w',      50.0,  True,    0.0,        None, None),
                     ('tau',    50.0,  True,    0.0,        None, None)
                     )
        testfitter2.add_analytical(
            'flipped_egh',
            testpars,
            funcname='test fEGH')
        testfitter2.perform_fit()
        testfitter3 = Fitter.fromspectrum(
            testspec2,
            psf=testpsf2)
        testpars = Parameters()
        #            (Name,    Value,  Vary,   Min,    Max,     Expr)
        testpars.add_many(
                     ('peak',   1.0,     True,   0.0,        None, None),
                     ('pos',    3000., True,   3000.-200.,  3000.+200., None),
                     ('fwhm',   50.0,  True,    0.0,        None, None),
                     )
        testfitter3.add_analytical(
            'gaussian',
            testpars,
            funcname='test gaussian')
        testfitter3.perform_fit()


class TestFitterResults:
    def test_fitres(self):
        """
        Test the returning of fit results
        """
        testspec = generate_cdespectrum()
        testfitter = Fitter(testspec.x.value, 2.*testspec.y.value)
        testpars = Parameters()
        #                 (Name,  Value,  Vary,   Min,     Max,     Expr)
        testpars.add_many(('mul', 1.0,    True,   0.0,     None,    None))
        testfitter.add_empirical(testspec, testpars)
        testfitter.perform_fit()
        testfitter.fit_results()

    def test_fitres_tofile(self):
        """
        Test the dumping of fit results to a file
        """
        testspec = generate_cdespectrum()
        testfitter = Fitter(testspec.x.value, 2.*testspec.y.value)
        testpars = Parameters()
        #                 (Name,  Value,  Vary,   Min,     Max,     Expr)
        testpars.add_many(('mul', 1.0,    True,   0.0,     None,    None))
        testfitter.add_empirical(testspec, testpars)
        testpars = Parameters()
        #        (Name,    Value,  Vary,   Min,    Max,     Expr)
        testpars.add_many(
                 ('peak',   1.0,     True,   0.0,        None, None),
                 ('pos',    3000., True,   3000.-200.,  3000.+200., None),
                 ('fwhm',   50.0,  True,    0.0,        None, None),
                 )
        testfitter.add_analytical(
            'gaussian',
            testpars,
            funcname='test gaussian')
        testfitter.perform_fit()
        testfitter.fitresults_tofile('testfile')

    def test_plotfit(self):
        """
        Test the plotting of fit results
        """
        testspec = generate_cdespectrum()
        testfitter = Fitter(testspec.x.value, 2.*testspec.y.value)
        testpars = Parameters()
        #                 (Name,  Value,  Vary,   Min,     Max,     Expr)
        testpars.add_many(('mul', 1.0,    True,   0.0,     None,    None))
        testfitter.add_empirical(testspec, testpars)
        testfitter.perform_fit()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        testfitter.plot_fitresults(ax)
        plt.close()
