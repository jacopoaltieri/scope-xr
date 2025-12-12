.. image:: https://github.com/jacopoaltieri/scope-xr/raw/main/src/scopexr/scopexr_logo.png
   :width: 300
   :align: center
   :alt: SCOPE-XR Logo

SCOPE-XR Documentation
======================

**SCOPE-XR (Single-image Characterization Of PErformance in X-Ray systems)** is a specialized Python framework for the automated characterization of X-ray systems. 

By analyzing a single acquisition of a circular aperture or disk test object, the software reconstructs 2D source distributions (Focal Spot) and detector responses (PSF).

.. note::
   This project implements methodologies by Di Domenico et al. and Forster et al. to provide high-precision X-ray diagnostics.

Key Features
------------
* **Focal Spot Analysis:** Automated 2D reconstruction of focal spot dimensions.
* **PSF Estimation:** 2D detector Point Spread Function analysis with sub-pixel oversampling.
* **Dual Interface:** Interactive GUI for routine analysis and CLI for batch research pipelines.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: Methodology

   theory
   pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`