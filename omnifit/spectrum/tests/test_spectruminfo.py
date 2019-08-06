# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ...tests.helpers import generate_cdespectrum


class TestSpectrumInfo:
    def test_minmax(self):
        """
        Make sure that min and max work
        """
        testspec = generate_cdespectrum()
        testspec.min()
        testspec.min(checkrange=[2000., 2050.])
        testspec.max()
        testspec.max(checkrange=[2000., 2050.])

    def test_info(self):
        """
        Make sure that spectrum info is produced
        """
        testspec = generate_cdespectrum()
        testspec.info()
