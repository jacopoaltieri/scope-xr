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


Citation
--------

**Published paper:**

The SCOPE-XR methodology and software are described in our paper; please cite it as:
Altieri J, Cardarelli P, Di Domenico G, Taibi A. A python framework for single-image characterization of X-ray focal spot distribution and detector point spread function. Med Phys. 2026;53:e70513. https://doi.org/10.1002/mp.70513




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