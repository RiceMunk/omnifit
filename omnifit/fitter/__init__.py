# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    # Not guaranteed available at setup time
    from .fitter import fitter
except ImportError:
    if not _ASTROPY_SETUP_:
        raise
