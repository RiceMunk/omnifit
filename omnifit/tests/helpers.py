# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
import os
import sys
from .. import spectrum
from .. import utils
epsilon = 1.e-10  # tolerance for floating point errors
sys._called_from_test = True


def generate_spectrum():
    """
    Generate and return a generic spectrum from dummy data
    """
    xdata = np.arange(1000, 2000, 10)*u.micron
    ydata = np.arange(0, 100, 1)*utils.unit_od
    return spectrum.BaseSpectrum(xdata, ydata)


def generate_absspectrum():
    """
    Generate and return an absorption spectrum suitable for testing
    """
    filepath_waterice = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir,
        'data/waterice_absorption.txt')
    wn, absorbance = np.loadtxt(
        filepath_waterice,
        delimiter=', ',
        skiprows=0,
        unpack=True)
    return spectrum.AbsorptionSpectrum(
        wn*u.kayser,
        absorbance*np.log(10)*utils.unit_od,
        specname='test water spectrum (absorption)')


def generate_absspectrum_alt():
    """
    Generate and return an absorption spectrum suitable for testing
    alternate version using a different set of data
    """
    filepath_watermeth = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir,
        'data/watermethanol_absorption.txt')
    wn, absorbance = np.loadtxt(filepath_watermeth, skiprows=0, unpack=True)
    return spectrum.AbsorptionSpectrum(
        wn*u.kayser,
        absorbance*np.log(10)*utils.unit_od,
        specname='test water spectrum (absorption)')


def generate_cdespectrum():
    """
    Generate and return a CDE spectrum (from n and k) suitable for testing
    """
    filepath_waterice = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir,
        'data/waterice_nandk.txt')
    wn, n, k, dum1, dum2 = np.loadtxt(
        filepath_waterice,
        skiprows=1,
        unpack=True)
    return spectrum.CDESpectrum(
        wn,
        np.vectorize(complex)(n, k),
        specname='test water spectrum (n and k)')
