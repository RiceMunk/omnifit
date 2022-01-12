import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import convolution
import warnings
import pickle
import os
from .. import utils
from copy import deepcopy
from functools import wraps


def clonable(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            clone = kwargs.pop('clone')
        except KeyError:
            clone = False
        if clone:
            newclass = deepcopy(self)
            retdata = getattr(newclass, func.__name__)(*args, **kwargs)
            if retdata is not None:
                return newclass, retdata
            else:
                return newclass
        else:
            retdata = func(self, *args, **kwargs)
            if retdata is not None:
                return retdata
    return wrapper


class BaseSpectrum:
    """
    A class to represent spectroscopic data.

    This class is designed to work for spectroscopic data of ices, but
    may work for other types of spectroscopic data as well.
    This is the most basic version of the class, concerned solely with
    the contents of the x and y attributes.

    Attributes
    ----------
    x : `astropy.units.Quantity`
        Represents the data on the "x-axis" of the spectrum,
        i.e. usually the wavelength or frequency
    y : `astropy.units.Quantity`
        Represents the data on the "x-axis" of the spectrum,
        i.e. the flux or optical depth
    dy : `NoneType` or `float`
        The uncertainty of y. Can be given during initialisation,
        or automatically calculated during baselining. (default=None)
    specname : `string`
        The name of the spectrum (default='Unknown spectrum')
    baselined : `bool`
        Indicates whether the spectrum has been baselined or not
    convolved : `bool`
        Indicates whether the spectrum has been put through convolution
    """
    def __init__(
            self,
            x,
            y,
            dy=None,
            specname='Unknown spectrum',
            nondata=[]):
        """
        BaseSpectrum(x,y,dy=None,specname='Unknown spectrum',nondata=[])

        Constructor for the BaseSpectrum class. Requires x and y;
        everything else is optional.

        Parameters
        ----------
        x : `astropy.units.Quantity` or `numpy.ndarray`
            Represents the data on the "x-axis" of the spectrum.
            This is stored as an astropy quantity and thus it is
            recommended that the class constructor is called with
            such an input. However, the constructor also accepts
            a numpy ndarray, in which case it will try to guess
            the units and then convert the input into an appropriate
            astropy quantity.
            The autodetection assumes the units are in kayser units
            (i.e. reciprocal wavenumbers with the unit cm^-1) if the
            mean of the input array is greater than 1000. Otherwise
            the autodetection assumes the units are in microns.
        y : `astropy.units.Quantity` or `numpy.ndarray`
            Represents the data on the "x-axis" of the spectrum.
            This is stored as an astropy quantity and thus it is
            recommended that the class constructor is called with
            such an input. However, the constructor also accepts
            a numpy ndarray, in which case it will assume that
            the units are in optical depth and then convert the
            input into this astropy quantity.
        dy : `float`, optional
            The uncertainty of y. If given, this is assumed to be
            the uncertainty of the y axis data in the same units
            as given (or assumed) with the y input. Otherwise
            the uncertainty is left as None during initialisation
            and will be calculated as part of baselining.
        specname : `string`, optional
            An optional human-readable name can be given to the
            spectrum via this input.
        nondata : `list`, optional
            If information unrelated to the x and y input data is
            stored in the class instance, the variable names in which
            this information is stored can be given here. This causes
            various internal functions (related to automatic sorting
            and error-checking) of the class to ignore these
            variables.
            It is not usually necessary for the user to use this input
            during initialisation; it is most often used by children of
            the BaseSpectrum class.
        """
        if len(x) != len(y):  # Check that input is sane
            raise RuntimeError('Input arrays have different sizes.')
        if type(x) != u.quantity.Quantity:
            # try to guess x units (between micron and kayser;
            # the most common units) if none given
            if np.mean(x) > 1000.:
                warnings.warn(
                    'The x data is not in astropy unit format. \
                    Autodetection assumes kayser.',
                    RuntimeWarning)
                self.x = x*u.kayser
            else:
                warnings.warn(
                    'The x data is not in astropy unit format. \
                    Autodetection assumes micron.',
                    RuntimeWarning)
                self.x = x*u.micron
        else:
            self.x = x
        if type(y) != u.quantity.Quantity:
            warnings.warn(
                'The y data is not in astropy unit format. \
                Assuming optical depth.',
                RuntimeWarning)
            self.y = y*utils.unit_od
        else:
            self.y = y
        if dy is not None:
            self.dy = dy
        else:
            self.dy = None
        self.name = str(specname)  # Spectrum name
        self.baselined = False     # Has the spectrum been baselined?
        self.convolved = False     # Has the spectrum been convolved?
        self.__nondata = [
            '_BaseSpectrum__nondata',
            'name',
            'convolved',
            'baselined',
            'dy']
        for cnondata in nondata:  # Extra non-array variable names into nondata
            if cnondata not in self.__nondata:
                self.__nondata.append(cnondata)
        self.__fixbad()  # Drop bad data
        self.__sort()

    def __sort(self):
        """
        __sort()

        An internal method which sorts the data arrays so that they
        all go in increasing order of x.

        Parameters
        ----------
        None
        """
        sorter = np.argsort(self.x)
        nondatavars = self.__nondata
        ownvarnames = self.__dict__.keys()
        ownvarnames = [
            i for i in
            filter(lambda a: a not in nondatavars, ownvarnames)
            ]
        for cVarname in ownvarnames:
            self.__dict__[cVarname] = self.__dict__[cVarname][sorter]

    def __fixbad(self):
        """
        __fixbad()

        An internal method which replaces all non-number data (e.g.
        infinities) in the data arrays with `numpy.nan`.

        Parameters
        ----------
        None
        """
        ignorevars = self.__nondata
        ownvarnames = self.__dict__.keys()
        ownvarnames = [
            i for i in
            filter(lambda a: a not in ignorevars, ownvarnames)
            ]
        varlength = len(self.__dict__[ownvarnames[0]])
        iGoodones = np.isfinite(np.ones(varlength))
        for cVarname in ownvarnames:
            cVar = self.__dict__[cVarname]
            if len(cVar) != varlength:
                raise RuntimeError(
                    'Anomalous variable length detected in spectrum!')
            iGoodones = np.logical_and(iGoodones, np.isfinite(cVar))
        iBadones = np.logical_not(iGoodones)
        for cVarname in ownvarnames:
            if cVarname != 'x':
                self.__dict__[cVarname][iBadones] = np.nan

    def plot(self, axis, x='x', y='y', **kwargs):
        """
        plot(axis, x='x', y='y', **kwargs)

        Plot the contents of the spectrum into a given matplotlib axis.
        Defaults to the data contained in the x and y attributes, but
        can also plot other data content if instructed to do so.

        Parameters
        ----------
        axis : `matplotlib.axis`
            The axis which the plot will be generated in.
        x : `string`, optional
            The name of the variable to be plotted on the x axis.
        y : `string`, optional
            The name of the variable to be plotted on the x axis.
        **kwargs : Arguments, optional
            This can be used to pass additional arguments
            to `matplotlib.pyplot.plot`, which is used by this
            method for its plotting.
        """
        try:  # assume it's with astropy units
            plotx = self.__dict__[x].value
        except ValueError:
            plotx = self.__dict__[x]
        try:  # assume it's with astropy units
            ploty = self.__dict__[y].value
        except ValueError:
            ploty = self.__dict__[y]
        axis.plot(plotx, ploty, **kwargs)

    @clonable
    def convert2wn(self):
        """
        convert2wn(clone=False)

        Convert the x axis data to kayser (reciprocal wavenumber) units.
        Re-sort the data afterwards.

        Parameters
        ----------
        clone : `bool`, optional
            If set to True, returns a modified copy of the spectrum instead
            of operating on the existing spectrum.
        """
        self.convert2(u.kayser)

    @clonable
    def convert2wl(self):
        """
        convert2wl(clone=False)

        Convert the x axis data to wavelength (in microns) units.
        Re-sort the data afterwards.

        Parameters
        ----------
        clone : `bool`, optional
            If set to True, returns a modified copy of the spectrum instead
            of operating on the existing spectrum.
        """
        self.convert2(u.micron)

    @clonable
    def convert2(self, newunit):
        """
        convert2(newunit, clone=False)

        Convert the x axis data to given spectral units.
        Re-sort the data afterwards.

        Parameters
        ----------
        newunit : `astropy.units.core.Unit`
            Desired (spectral) unit the x axis data should be
            converted to.
        clone : `bool`, optional
            If set to True, returns a modified copy of the spectrum instead
            of operating on the existing spectrum.
        """
        with u.set_enabled_equivalencies(u.equivalencies.spectral()):
            self.x = self.x.to(newunit)
        self.__sort()

    @clonable
    def subspectrum(self, limit_lower, limit_upper):
        """
        subspectrum(limit_lower, limit_upper, clone=False)

        Cropped the spectrum along along the x axis using the given
        inclusive limits.

        Parameters
        ----------
        limit_lower : `float`
            The desired minimum x axis of the cropped spectrum, in
            current units of the spectrum. This limit is inclusive.
        limit_upper : `float`
            The desired maximum x axis of the cropped spectrum, in
            current units of the spectrum. This limit is inclusive.
        clone : `bool`, optional
            If set to True, returns a modified copy of the spectrum instead
            of operating on the existing spectrum.
        """
        iSub = np.logical_and(
            np.greater_equal(self.x.value, limit_lower),
            np.less_equal(self.x.value, limit_upper))
        newX = self.x[iSub]
        newY = self.y[iSub]
        self.x = newX
        self.y = newY

    @clonable
    def interpolate(self, target_spectrum):
        """
        interpolate(target_spectrum,clone=False)

        Interpolate spectrum to match target spectrum resolution.
        Does not modify current spectrum, but replaces it with a new one,
        which is a copy of the current spectrum but with the interpolated
        data on the x and y fields.
        The target spectrum has to be using the compatible units on the
        x and y axes as the current spectrum, or the interpolation will fail
        (including, e.g., units of wavenumbers/frequency/wavelength).

        Parameters
        ----------
        target_spectrum : `BaseSpectrum`
            The target spectrum which the x axis resolution of the current
            spectrum should be made to match.
        clone : `bool`, optional
            If set to True, returns a modified copy of the spectrum instead
            of operating on the existing spectrum.
        """
        if not self.x.unit.is_equivalent(
                target_spectrum.x.unit, equivalencies=u.spectral()):
            raise u.UnitsError('Spectra have incompatible units on x axis!')
        if not self.y.unit.is_equivalent(
                target_spectrum.y.unit, equivalencies=u.spectral_density(self.x)):
            raise u.UnitsError('Spectra have incompatible units on y axis!')
        newX = target_spectrum.x.to(self.x.unit, equivalencies=u.spectral())
        newY = np.interp(newX, self.x, self.y)
        self.x = newX
        self.y = newY
        self.name = '{0}(interpolated: {1})'.format(
            self.name,
            target_spectrum.name)

    def yat(self, x):
        """
        yat(x)

        Return interpolated value of y at requested x.

        Parameters
        ----------
        x : `float`
            The x axis coordinate of interest.

        Returns
        -------
        The interpolated value of y at the requested x coordinate.

        """
        return np.interp(x, self.x.value, self.y.value)

    @clonable
    def convolve(self, kernel, **kwargs):
        """
        convolve(kernel, clone=False, **kwargs)

        Use `astropy.convolution.convolve` to convolve the y axis data of the
        spectrum with the given kernel.

        Parameters
        ----------
        kernel : `numpy.ndarray` or `astropy.convolution.Kernel`
            A convolution kernel to feed into the convolution function.
        clone : `bool`, optional
            If set to True, returns a modified copy of the spectrum instead
            of operating on the existing spectrum.
        **kwargs : Arguments, optional
            This can be used to pass additional arguments
            to `astropy.convolution.convolve`.
        """
        if self.convolved:
            warnings.warn(
                'Spectrum '+self.name+' has already been convolved once!',
                RuntimeWarning)
        self.y = convolution.convolve(self.y, kernel, **kwargs)
        self.convolved = True

    @clonable
    def gconvolve(self, fwhm, **kwargs):
        """
        gconvolve(fwhm,**kwargs)

        Convolve spectrum using a gaussian of given fwhm.

        Parameters
        ----------
        fwhm : `float`
            The desired fwhm of the gaussian, in units of x axis.
        **kwargs : Arguments, optional
            This can be used to pass additional arguments
            to `convolve`.
        """
        gkernel = convolution.Gaussian1DKernel(fwhm)
        self.convolve(gkernel, **kwargs)

    @clonable
    def smooth(self, window_len=11, window='hanning'):
        """
        smooth(window_len=11, window='hanning', clone=False)

        Smooth the spectrum using the given window of requested type and size.
        The supported smoothing functions are: Bartlett, Blackman, Hanning,
        Hamming, and flat (i.e. moving average).
        This method has been adapted from http://stackoverflow.com/a/5516430

        Parameters
        ----------
        window_len : `int`, optional
            Desired window size, in increments of x axis.
        window : {'flat','hanning','hamming','bartlett','blackman'}, optional
            Desired window type.
        clone : `bool`, optional
            If set to True, returns a modified copy of the spectrum instead
            of operating on the existing spectrum.
        """
        VALID_WINDOWS = [
            'flat',
            'hanning',
            'hamming',
            'bartlett',
            'blackman']
        if self.x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        if self.x.size < window_len:
            raise ValueError(
                "Input vector needs to be bigger than window size.")
        if window_len < 3:
            self.y = self.x
        else:
            if window not in VALID_WINDOWS:
                raise ValueError(
                    "Window must be one of the following: {}".format(
                        ','.join(VALID_WINDOWS)))
            s = np.r_[
                2*self.x[0]-self.x[window_len-1::-1],
                self.x,
                2*self.x[-1]-self.x[-1:-window_len:-1]
                ]
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.'+window+'(window_len)')
            self.y = np.convolve(w/w.sum(), s, mode='same')

    def baseline(
            self,
            degree=1,
            windows=[[0.0, 1.0e6]],
            exclusive=False,
            usefile=None):
        """
        baseline(
            degree=1,
            windows=[[0.0,1.0e6]],
            exclusive=False,
            usefile=None)

        Fit and subtract a polynomial baseline from the spectrum, within
        the specified windows. The fitting windows can either be designated
        as a list of x axis coordinates, or specified interactively within
        a matplotlib plotting window.

        Parameters
        ----------
        degree : `int`, optional
            Degree of order on the polynomial to fit.
        windows : `list` or `string`, optional
            The windows can be designated in two different ways:

            * as a list of x axis coordinates, e.g.
                [[200,250],[300,350]] for two windows
                of 200 to 250, and 300 to 350.
            * in an interactive matplotlib plotting window, by
                setting windows to 'manual'

            In the former case, no further input is required from
            the user after calling baseline, but in the latter case
            the baseliner class is invoked from omnifit.utils.
        exclusive : `bool`, optional
            This parameter indicates whether the windows are exclusive
            or inclusive, i.e. whether the polynomial baseline fitting
            is done inside (exclusive=False) the range indicated by
            the windows or outside (exclusive=True) of said range.
        usefile : `NoneType` or `string`
            This parameter indicates whether an interactively designated
            baseline data is saved into a file, or if the baseline data
            is read from an already existing file.
            If the user wishes to use an existing file for a baseline,
            simply set usefile as the path to the pickle file created
            in a previous baselining session.
            To create a new baseline file, set windows to 'manual' and
            set usefile to point to the desired path of the new file.
        """
        iBaseline = np.logical_or(np.isinf(self.x), exclusive)
        if usefile is not None and os.path.exists(usefile):
            with open(usefile, 'r') as cFile:
                windows = pickle.load(cFile)
        elif windows == 'manual':
            print('Determining manual baseline')
            cFig = plt.figure()
            cAx = cFig.add_subplot(111)
            cManager = plt.get_current_fig_manager()
            cManager.window.wm_geometry("+100+50")
            cAx.plot(self.x, self.y, 'k-', drawstyle='steps-mid')
            cBaseliner = utils.Baseliner(cAx, self)
            plt.show(cFig)
            windows = cBaseliner.windows
            if usefile is not None:
                with open(usefile, 'w') as cFile:
                    pickle.dump(windows, cFile)
                print('Wrote window data to '+usefile)
        for cWindow in windows:
            if exclusive:
                iBaseline = np.logical_and(
                    iBaseline,
                    np.logical_or(
                        np.less(self.x.value, cWindow[0]),
                        np.greater(self.x.value, cWindow[1])))
            else:
                iBaseline = np.logical_or(
                    iBaseline,
                    np.logical_and(
                        np.greater(self.x.value, cWindow[0]),
                        np.less(self.x.value, cWindow[1])))
        baseline = np.polyfit(
            self.x.value[iBaseline],
            self.y.value[iBaseline],
            degree)
        if not(np.all(np.isfinite(baseline))):
            raise RuntimeError('Baseline is non-finite!')
        fixedY = self.y.value
        for cDegree in range(degree+1):
            fixedY = fixedY-baseline[degree-cDegree]*self.x.value**cDegree
        self.y = fixedY*self.y.unit
        if self.dy is None:
            self.dy = np.abs(np.std(fixedY[iBaseline]))
        self.baselined = True

    def shift(self, amount):
        """
        shift(amount)

        Shifts the spectrum by amount
        specified, in primary x axis
        units.

        Parameters
        ----------
        x : `float`
            The x axis of the entire spectrum has this number
            added to it, effectively shifting it.
        """
        self.x += amount

    def max(self, checkrange=None):
        """
        max(checkrange=None)

        Returns maximum y of the spectrum.
        If checkrange is set, returns maximum inside of that range.

        Parameters
        ----------
        `checkrange` : `Nonetype` or `list`
            If this is set to a list, the first and second items on
            the list are taken to indicate the range (in units of x axis)
            between which the maximum is looked for.

        Returns
        -------
        Maximum y of either the entire spectrum or, if checkrange is set,
        the maximum y inside of the specified range.
        """
        iCheckrange = np.ones_like(self.y.value, dtype=bool)
        if checkrange is not None:
            minX = checkrange[0]
            maxX = checkrange[1]
            iCheckrange = np.logical_and(
                iCheckrange,
                np.logical_and(
                    np.less_equal(minX, self.x.value),
                    np.greater_equal(maxX, self.x.value)))
        return np.nanmax(self.y[iCheckrange])

    def min(self, checkrange=None):
        """
        min(checkrange=None)

        Returns minimum y of the spectrum.
        If checkrange is set, returns minimum inside of that range.

        Parameters
        ----------
        checkrange : `Nonetype` or `list`
            If this is set to a list, the first and second items on
            the list are taken to indicate the range (in units of x axis)
            between which the minimum is looked for.

        Returns
        -------
        Minimum y of either the entire spectrum or, if checkrange is set,
        the minimum y inside of the specified range.
        """
        iCheckrange = np.ones_like(self.y.value, dtype=bool)
        if checkrange is not None:
            minX = checkrange[0]
            maxX = checkrange[1]
            iCheckrange = np.logical_and(
                iCheckrange,
                np.logical_and(
                    np.less_equal(minX, self.x.value),
                    np.greater_equal(maxX, self.x.value)))
        return np.nanmin(self.y[iCheckrange])

    def info(self):
        """
        info()

        Prints out a simple human-readable summary of the spectrum,
        containing the name of the spectrum, the units on its axes,
        and their limits. Also shows whether the spectrum has been
        baselined or convolved yet.

        Parameters
        ----------
        None

        Returns
        -------
        Nothing, but prints out a summary of the spectrum.
        """
        print('---')
        print('Summary for spectrum '+self.name)
        print('x unit: '+str(self.x.unit))
        print('min(x): '+str(np.nanmin(self.x.value)))
        print('max(x): '+str(np.nanmax(self.x.value)))
        print('y unit: '+str(self.y.unit))
        print('min(y): '+str(np.nanmin(self.y.value)))
        print('max(y): '+str(np.nanmax(self.y.value)))
        print('baselined: '+str(self.baselined))
        print('convolved: '+str(self.convolved))
        print('---')


class AbsorptionSpectrum(BaseSpectrum):
    """
    A class specialized in representing absorption spectra of the type
    often used in ice spectroscopy.

    The functionality of this class is otherwise identical to `BaseSpectrum`,
    except it contains an additional method for plotting the optical depth
    spectrum in either microns and kayser units, both of which it stores
    as additional attributes. Upon initialisation it uses the kayser units
    as its x axis units.

    Attributes
    -----------
    All the attributes stored in `BaseSpectrum`, plus the following:
    wn : `astropy.units.Quantity`
        The data on the x-axis of the spectrum, expressed in kayser
        (reciprocal wavenumber) units.
    wl : `astropy.units.Quantity`
        The data on the x-axis of the spectrum, expressed in microns.
    od : `astropy.units.Quantity`
        The data on the y-axis spectrum, expressed as optical depth
        units (using omnifit.utils.unit_od).
    """
    def __init__(self, wn, od, **kwargs):
        """
        AbsorptionSpectrum(wn,od,**kwargs)

        Constructor for the `AbsorptionSpectrum` class.

        Parameters
        ----------
        wn : `astropy.units.Quantity`
            The absorption spectrum frequency data. Unlike `BaseSpectrum`,
            the initialisation of `AbsorptionSpectrum` requires this to be
            in the specific units of reciprocal wavenumber. However, if it is
            in a quantity convertable to kayser, conversion will be attempted
            while a warning is given to notify the user of this.
        od : `astropy.units.Quantity`
            The absorption spectrum optical depth data. Unlike `BaseSpectrum`,
            the initialisation of `AbsorptionSpectrum` requires this to be
            in the specific units of optical depth units (from
            `omnifit.utils.unit_od`).
        **kwargs : Arguments, optional
            Additional initialisation arguments can be passed to `BaseSpectrum`
            using this. Note that x and y are defined using the other
            initialisation parameters of `AbsorptionSpectrum`.
        """
        if type(wn) != u.quantity.Quantity:
            raise u.UnitsError('Input wn is not an astropy quantity.')
        if wn.unit != u.kayser:
            warnings.warn(
                'Input wn is not in kayser units. Converting...',
                RuntimeWarning)
            with u.set_enabled_equivalencies(u.equivalencies.spectral()):
                wn = wn.to(u.kayser)
        if type(od) != u.quantity.Quantity:
            raise u.UnitsError('Input od is not an astropy quantity.')
        if od.unit != utils.unit_od:
            raise u.UnitsError('Input od is not in optical depth units.')
        if len(wn) != len(od):
            raise RuntimeError('Input arrays have different sizes.')
        self.wn = wn
        with u.set_enabled_equivalencies(u.equivalencies.spectral()):
            self.wl = self.wn.to(u.micron)
        self.od = od
        BaseSpectrum.__init__(self, self.wn, self.od, **kwargs)

    def plotod(self, ax, in_wl=False, **kwargs):
        """
        plotod(ax, in_wl=False, **kwargs)

        Plot the optical depth spectrum as either a function of reciprocal
        wavenumber or wavelength to the given axis.

        Parameters
        ----------
        axis : `matplotlib.axis`
            The axis which the plot will be generated in.
        in_wl : `bool`, optional
            If set to true, the x axis of the plotting axis will be in
            wavelength; otherwise it will be in reciprocal wavenumbers.
        **kwargs : Arguments, optional
            This can be used to pass additional arguments
            to `matplotlib.pyplot.plot`, which is used by this
            method for its plotting.
        """
        if in_wl:
            self.plot(ax, x='wl', y='od', **kwargs)
        else:
            self.plot(ax, x='wn', y='od', **kwargs)


class CDESpectrum(AbsorptionSpectrum):
    """
    A class specialized in representing CDE-corrected absorption spectra.

    The functionality of this class is otherwise identical to
    `AbsorptionSpectrum` (and by extension, `BaseSpectrum`), except it
    contains an additional method for plotting the complex refractive index
    data, which it also stores in additional attributes as part of the class
    instance. Also stored are various additional data returned by the CDE
    correction, as documented below.

    Attributes
    -----------
    All the attributes stored in `AbsorptionSpectrum`, plus the following:
    m : `numpy.ndarray`
        The complex refractive index spectrum of the data.
    cabs : `numpy.ndarray`
        The spectrum of the absorption cross section of the simulated grain.
    cabs_vol : `numpy.ndarray`
        The spectrum of the absorption cross section of the simulated grain,
        normalized by the volume distribution of the grain. This parameter
        is the equivalent of optical depth in most cases.
    cscat_vol : `numpy.ndarray`
        The spectrum of the scattering cross section of the simulated grain,
        normalized by the volume distribution of the grain.
    ctot : `numpy.ndarray`
        The spectrum of the total cross section of the simulated grain.
    """
    def __init__(self, wn, m, **kwargs):
        """
        CDESpectrum(wn, m, **kwargs)

        Constructor for the `CDESpectrum` class.

        Parameters
        ----------
        wn : `astropy.units.Quantity` or `numpy.ndarray`
            The absorption spectrum frequency data. If given as
            `astropy.units.Quantity`, they must either be in kayser (reciprocal
            wavenumbers) or convertable to kayser. If given as `numpy.ndarray`,
            they are assumed to be in kayser.
        m : `numpy.ndarray`
            The complex refractive index spectrum of the data.
        **kwargs : Arguments, optional
            Additional initialisation arguments can be passed to
            `AbsorptionSpectrum` using this. Note that x and y are defined
            using the other initialisation parameters of `CDESpectrum`.
        """
        if len(wn) != len(m):
            raise RuntimeError('Input arrays have different sizes.')
        if type(wn) != u.quantity.Quantity:
            wn = wn * u.kayser
        if wn.unit != u.kayser:
            with u.set_enabled_equivalencies(u.equivalencies.spectral()):
                wn = wn.to(u.kayser)
        self.cabs, self.cabs_vol, self.cscat_vol, self.ctot = \
            utils.cde_correct(wn.value, m)
        self.m = np.array(m, dtype=complex)
        od = self.cabs_vol*utils.unit_od
        AbsorptionSpectrum.__init__(self, wn, od, **kwargs)

    def plotnk(self, ax_n, ax_k, **kwargs):
        """
        plotnk(ax_n, ax_k, **kwargs)

        Plot the complex refractive indices as function of wavenumber to
        the two given matplotlib axes.

        Parameters
        ----------
        ax_n : `matplotlib.axis`
            The axis which the plot of n will be generated in.
        ax_k : `matplotlib.axis`
            The axis which the plot of k will be generated in.
        **kwargs : Arguments, optional
            This can be used to pass additional arguments
            to `matplotlib.pyplot.plot`, which is used by this
            method for its plotting.
        """
        ax_n.plot(self.wn, self.m.real, **kwargs)
        ax_k.plot(self.wn, self.m.imag, **kwargs)
