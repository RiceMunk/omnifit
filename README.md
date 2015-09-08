# Omnifit
https://ricemunk.github.io/omnifit/

Omnifit is a Python library originally created to easily facilitate multi-component fitting of laboratory spectra and analytical data to astronomical observational data of interstellar ices.

## Requirements
Omnifit has the following requirements (including their dependencies):
 * Python <http://www.python.org> 2.7 (support for 3.x is planned)
 * Numpy <http://www.numpy.org> (tested to work with 1.9.2)
 * lmfit <http://lmfit.github.io/lmfit-py/> 0.8.3 or later (tested to work with 0.8.3)
 * Astropy <http://astropy.org> 1.0 or later (tested to work with 1.0.3)

## Installation
### Using pip
Installation using pip works by running

  pip install --pre omnifit

### Building from source
First you must download the source code for omnifit.
The most up to date development version of omnifit can be cloned from the github repository <https://github.com/RiceMunk/omnifit>.

Once you have downloaded and extracted the Omnifit source files, you can build it by invoking the command

  python setup.py build

in the directory containing setup.py. After omnifit is finished building, you may install it by invoking

  python setup.py install

If you wish to install Omnifit on a computer on which you do not have administrator rights, you can instead use the command

  python setup.py install --user

After installation completes you should be able to import omnifit in Python using the command

  import omnifit

If this does not raise an error, omnifit has been successfully installed on your system!

##Citing Omnifit
The main paper showcasing Omnifit is currently in the final stages of preparation. In the meantime, it is possible to cite Omnifit using Zenodo, with the DOI 10.5281/zenodo.29354.
