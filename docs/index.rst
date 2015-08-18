.. Much of what is found below has been shamelessly copied from the astropy documentation layout.

.. the "raw" directive below is used to hide the title in favor of just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 { display:none; }
    </style>

#############################
Omnifit Package Documentation
#############################

.. |logo_svg| image:: _static/omnifit_banner.svg

.. |logo_png| image:: _static/omnifit_banner.png

.. raw:: html

   <img src="_images/omnifit_banner.svg" onerror="this.src='_images/omnifit_banner.png'; this.onerror=null;" width="485"/>

.. only:: latex

    .. image:: _static/omnifit_banner.pdf

Welcome to the Omnifit documentation!

************
Introduction
************
.. only:: html

    :doc:`whatsnew/0.1`
    -------------------

.. only:: latex

    .. toctree::
       :maxdepth: 1

       whatsnew/0.1

**Omnifit basics**

.. toctree::
  :maxdepth: 1

  overview
  examples

*************
Reference/API
*************
Spectrum
========
  .. automodapi:: omnifit.spectrum
Fitter
========
  .. automodapi:: omnifit.fitter
Utils
=====
  .. automodapi:: omnifit.utils