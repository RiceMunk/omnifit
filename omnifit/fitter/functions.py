import numpy as np
from astropy import convolution


def muldata(data, mul):
    """
    muldata(data, mul)

    Multiplies data with a given multiplier, and returns the result.

    Parameters
    ----------
    data : numpy.ndarray
        The data to multiply
    mul : float
        The multiplier to multiply the data with

    Returns
    -------
    The data multiplied by the multiplier.
    """
    return mul*data


def flipped_egh(x, par, kernel=None):
    """
    flipped_egh(x, par, kernel=None)

    A flipped EGH (exponential-gaussian hybrid) function. A type of
    lopsided Gaussian function.
    This has been adapted from a normal EGH, which is presented in
    Lan & Jorgenson, Journal of Chromatography A, 915 (2001) 1-13

    Parameters
    ----------
    x : numpy.ndarray
        The x axis data of function.
    par : lmfit.parameter.Parameters
        The lmfit Parameters instance, which should contain the following:
            * H - the magnitude of the peak maximum
            * xR - the "retention time" of the precursor Gaussian
            * w - the standard deviation of the precursor Gaussian
            * tau - the time constant of precursor exponential
    kernel : Nonetype, numpy.ndarray or astropy.convolution.Kernel
        If set, the result will be convolved using this kernel.

    Returns
    -------
    The function calculated over the range in x, and convolved with the
    kernel if one was given.
    """
    H = par['H'].value
    xR = par['xR'].value
    w = par['w'].value
    tau = par['tau'].value
    expFactor = np.exp((-1.0*(xR-x)**2.0)/(2.0*w*w+tau*(xR-x)))
    out_y = np.where(2.0*w*w+tau*(xR-x) > 0, H*expFactor, 0.0)
    if kernel is None:
        return out_y
    else:
        return convolution.convolve(out_y, kernel)


def gaussian(x, par, kernel=None):
    """
    gaussian(x, par, kernel=None)

    An 1D Gaussian function, structured in a way that it can be given an
    lmfit Parameters instance as input.

    Parameters
    ----------
    x : numpy.ndarray
        The x axis data of function.
    par : lmfit.parameter.Parameters
        The lmfit Parameters instance, which should contain the following:
            * peak - the peak height of the Gaussian
            * fwhm - the full width at half maximum of the Gaussian
            * pos - the peak position of the Gaussian
    kernel : Nonetype, numpy.ndarray or astropy.convolution.Kernel
        If set, the result will be convolved using this kernel.

    Returns
    -------
    The function calculated over the range in x, and convolved with the
    kernel if one was given.
    """
    peak = par['peak'].value
    fwhm = par['fwhm'].value
    pos = par['pos'].value
    out_y = peak*np.exp(-2.35*(x-pos)**2./fwhm**2.)
    if kernel is None:
        return out_y
    else:
        return convolution.convolve(out_y, kernel)


def cde_lorentzian(x, par, kernel=None):
    """
    cde_lorentzian(x, par, kernel=None)

    A CDE-corrected Lorentzian function, sometimes used for representing
    astrophysical ices. For more information about this fucntion and at
    least one application for it, see:
    K. Pontoppidan et al., Astronomy & Astrophysics, 408 (2003) 981-1007

    Parameters
    ----------
    x : numpy.ndarray
        The x axis data of function.
    par : lmfit.parameter.Parameters
        The lmfit Parameters instance, which should contain the following:
            * lor1 - the first Lorentzian parameter, describing the dielectric
                function at frequencies low relative to the peak described by
                this function.
            * lor2 - the second Lorentzian parameter, describing the plasma
                frequency.
            * lor3 - the third Lorentzian parameter, related to the mass and
                imagined spring constant of the molecule which it describes.
            * peak - the peak height of the function
            * pos - the peak position of the function
    kernel : Nonetype, numpy.ndarray or astropy.convolution.Kernel
        If set, the result will be convolved using this kernel.

    Returns
    -------
    The function calculated over the range in x, and convolved with the
    kernel if one was given.
    """
    lor1 = par['lor1'].value
    lor2 = par['lor2'].value
    lor3 = par['lor3'].value
    peak = par['peak'].value
    pos = par['pos'].value
    lorentz_oscillator = lor1+lor2**2./(pos**2.-x**2.-lor3*x*1.j)
    out_y = peak*x*np.imag(
        2.*lorentz_oscillator*np.log10(lorentz_oscillator) /
        (lorentz_oscillator-1.))
    if kernel is None:
        return out_y
    else:
        return convolution.convolve(out_y, kernel)
