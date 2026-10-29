Quickstart Guide
================

This guide will walk you through your first analysis with SCOPE-XR using the provided example images.

Prerequisites
-------------

Make sure you have SCOPE-XR installed. If not, see :doc:`installation`.

Your First Focal Spot Analysis
-------------------------------

SCOPE-XR includes default configuration files in the package and sample
images in the repository to help you get started.

Step 1: Get the configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After installation, export the default configuration files:

.. code-block:: bash

    scopexr-confex

This exports:

* ``fs_args.yaml`` - Focal spot configuration
* ``psf_args.yaml`` - PSF configuration

To place the configuration files in another directory, use:

.. code-block:: bash

    scopexr-confex --output-dir path/to/configs

Step 2: Get the example images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The sample images are available in the repository's ``examples/`` folder. You can download them directly from here:


* :download:`Download Focal Spot Sample <https://raw.githubusercontent.com/jacopoaltieri/scope-xr/main/examples/simulations/Image_FS2_Double_Asym_Clean.tif>`
* :download:`Download PSF Sample <https://raw.githubusercontent.com/jacopoaltieri/scope-xr/main/examples/simulations/Image_PSF_Asym_1.5px_Clean.tif>`

Step 3: Run the GUI
~~~~~~~~~~~~~~~~~~~~

The easiest way to start is with the graphical interface:

.. code-block:: bash

    scopexr-gui

**What happens next:**

1. The GUI opens with two tabs: "Focal Spot (FS)" and "PSF"
2. Click the **Focal Spot (FS)** tab
3. Click **"Browse"** to select ``Image_FS2_Double_Asym_Clean.tif``
4. The image preview appears in the GUI (PNG, TIFF, and DICOM/IMA images show previews)
5. Click **"Run Analysis"**
6. Watch the output in the console window
7. Results are saved to the output directory (default: ``output_fs/``)


Your First PSF Analysis
-----------------------

Step 1: Switch to PSF Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the GUI:

1. Click the **PSF** tab
2. Click **"Browse"** to select ``Image_PSF_Asym_1.5px_Clean.tif``
3. Review the automatically loaded parameters from ``psf_args.yaml``

Step 2: Run PSF Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Or use CLI:
    scopexr-psf --f "examples/Image_PSF_Asym_1.5px_Clean.tif"

CLI Quick Reference
-------------------

For automated workflows, use the command-line interface:

**Focal Spot:**

.. code-block:: bash

    # Basic usage
    scopexr-fs --f "path/to/image.tif"
    
    # Override pixel size
    scopexr-fs --f "image.tif" --p 0.15
    
    # Specify output directory
    scopexr-fs --f "image.tif" --o "my_results/"

**PSF:**

.. code-block:: bash

    # Basic usage
    scopexr-psf --f "path/to/image.tif"
    
    # With sub-pixel oversampling
    scopexr-psf --f "image.tif" --oversample --dtheta 2 --oversampling_factor 3

Common First-Time Issues
------------------------

**Issue: "No circular object detected"**

* **Solution**: Your image may have low contrast. Try adjusting Hough transform parameters in the YAML file or via the GUI, or skip circle detection entirely (if the circle is centered):
  
  .. code-block:: yaml
  
      hough_params:
        minRadius: 50
        maxRadius: 500

**Issue: "Magnification estimation failed"**

* **Solution**: Provide magnification manually:

  .. code-block:: bash
  
      scopexr-fs --f "image.tif" --m 1.5

**Issue: Results look noisy**

* **Solution**: Enable sinogram symmetrization:

  .. code-block:: bash
  
      scopexr-fs --f "image.tif" --sym

Next Steps
----------

* Read the full :doc:`usage` guide for all available parameters
* Understand the :doc:`theory` behind the measurements
* Review the :doc:`pipeline` for quality control
* See :doc:`troubleshooting` for common issues

For batch processing, export the bundled configurations or create custom
YAML configuration files and use the CLI in scripts:

.. code-block:: bash

    #!/bin/bash
    scopexr-confex --output-dir configs
    for img in data/*.tif; do
        scopexr-fs --f "$img" --config configs/fs_args.yaml
    done
