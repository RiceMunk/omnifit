Changelog
=========

Changes in 0.1.1
----------------
 * Added pypi support.
 * Added BSD license notification.

Changes in 0.1
--------------
 * This is the first public release version of Omnifit.
 * Most spectrum manipulation methods now have an option of returning a new spectrum instead of modifying the existing spectrum.
 * Arbitrary arguments can now be passed from the constructors of child classes to their parents.
 * It is now possible to give arbitrary matplotlib arguments to the various plotting methods used by Omnifit, and the amount of 'hard-coded' functionality has been minimized in these methods.
 * The spectrum classes now preferentially make use of Astropy quantity arrays instead of Numpy arrays to store their data. Also the unit conversions are now performed by the Astropy functions related to unit conversions, instead of the limited number of conversions previously supported by Omnifit.
 * Convolution now makes use of Astropy convolution routines instead of Numpy convolution routines.
 * Sphinx documentation added.
 * Numerous minor bugfixes.
