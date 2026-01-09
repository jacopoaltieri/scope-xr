.. image:: https://github.com/jacopoaltieri/scope-xr/raw/main/src/scopexr/scopexr_logo.png
   :width: 300
   :align: center
   :alt: SCOPE-XR Logo

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
.. image:: https://img.shields.io/badge/GitHub-Repository-lightgrey?logo=github
   :target: https://github.com/jacopoaltieri/scope-xr

SCOPE-XR Documentation
======================

**SCOPE-XR (Single-image Characterization Of PErformance in X-Ray systems)** is a specialized Python framework for the automated characterization of X-ray systems. 
The source code is hosted on `GitHub <https://github.com/jacopoaltieri/scope-xr>`_.

By analyzing a single acquisition of a circular aperture or disk test object, the software reconstructs 2D source distributions (Focal Spot) and detector response (PSF).


Key Features
------------
* **Focal Spot Analysis:** Automated 2D reconstruction of focal spot dimensions.
* **PSF Estimation:** 2D detector Point Spread Function analysis with sub-pixel oversampling.
* **Dual Interface:** Interactive GUI for routine analysis and CLI for batch research pipelines.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   usage
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Methodology

   theory
   pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`