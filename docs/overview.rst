Overview
========
Omnifit is a Python library originally created to easily facilitate multi-component fitting of laboratory spectra and analytical data to astronomical observational data of interstellar ices. A research paper showcasing the capabilities of an early pre-release version of Omnifit is currently in final stages of preparation, and will be linked to here when it is published.

What is Omnifit?
****************
At its heart Omnifit is simply a convenience tool intended to make a specific type of spectral analysis (that of interstellar ices) as easy as possible.

You may have a reason to use Omnifit if you:

 * Have spectroscopic data which you suspect contains signs of ices.
 * You have either additional spectroscopic data which you want to try and fit to your observed data or you know how to describe the shape of the suspected ice feature with an analytical function.

Omnifit is useful over simply combining your observations and model(s) using your favorite minimization algorithm because of a few factors:

 * It is easy to make Omnifit combine an arbitrary number of analytical functions and empirically acquired spectra to fit against the target data, without having to spend time constructing a residual function which takes all the fitting parameters in one chunk. Instead you can manage separate functions and sets of data as their own instances and easily add or remove them from the total fit, without having to rewrite your whole function.
 * Omnifit has the tools for easily performing a number of pre-processing steps for the spectroscopic data, such as calculating the optical constants from a raw laboratory spectrum, interpolating one spectrum to match the resolution of another spectrum, applying smoothing functions to the spectral data, or extracting a sub-spectrum from a larger spectrum.