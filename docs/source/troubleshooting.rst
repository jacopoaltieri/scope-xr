Troubleshooting
===============

This guide covers common issues and their solutions when using SCOPE-XR.

Installation Issues
-------------------

Package Installation Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``pip install`` fails with dependency errors.

**Solutions:**

1. Upgrade pip:

   .. code-block:: bash

       pip install --upgrade pip

2. Install in a fresh virtual environment:

   .. code-block:: bash

       python -m venv fresh_env
       # Windows: .\fresh_env\Scripts\activate
       # Linux/macOS: source fresh_env/bin/activate
       pip install git+https://github.com/jacopoaltieri/scope-xr.git

3. If OpenCV fails, try the headless version:

   .. code-block:: bash

       pip install opencv-python-headless

PyQt6 Installation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** GUI won't start due to PyQt6 errors.

**Solutions:**

* On Linux, install system dependencies:

  .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get install python3-pyqt6
      
      # Fedora
      sudo dnf install python3-qt6

* Try PyQt5 as alternative (requires code modification)

Image Loading Issues
--------------------

"File not found" Error
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Cannot load image file.

**Checklist:**

1. Verify file path is correct (use absolute paths)
2. Check file extension is ``.png``, ``.tif``, or ``.raw``
3. For RAW files, ensure corresponding ``.xml`` metadata file exists
4. Check file permissions

RAW Image Format Not Recognized
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** RAW file loads but produces errors.

**Requirements for RAW files:**

* Must have accompanying XML metadata file with same name
* XML must contain: ``<ImageWidth>``, ``<ImageHeight>``
* Raw data must be in little-endian format

**Example XML structure:**

.. code-block:: xml

    <?xml version="1.0"?>
    <Image>
        <ImageWidth>2048</ImageWidth>
        <ImageHeight>2048</ImageHeight>
        <BitsPerPixel>16</BitsPerPixel>
    </Image>

TIF Images Appear Black
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Image loads but appears completely black.

**Solutions:**

1. Check bit depth - SCOPE-XR supports 8-bit and 16-bit images
2. Verify image is not inverted (dark background required)
3. Check that pixel values aren't all zeros (corrupted file)
4. Try opening in ImageJ/FIJI to verify image integrity

Circle Detection Problems
--------------------------

"No circular object detected"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Hough Transform fails to find the aperture.

**Solutions:**

1. **Adjust Hough parameters** in YAML config or via GUI:

   .. code-block:: yaml

       hough_params:
         minRadius: 30        # Reduce if object is small
         maxRadius: 800       # Increase if object is large
         dp: 1.5              # Increase for faster, coarser search
         param1: 50           # Lower for low contrast
         param2: 30           # Lower for incomplete circles

2. **Increase image contrast** before analysis
3. **Check image orientation** - circle must be clearly visible
4. **Use manual center selection** (bypass Hough):

   .. code-block:: bash

       scopexr-fs --f "image.tif" --no_hough

Multiple Circles Detected
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Program detects multiple circles or wrong circle.

**Solutions:**

* Increase ``param2`` in Hough parameters (makes detection stricter)
* Crop image to show only the test object
* Set tighter ``minRadius`` and ``maxRadius`` constraints

Reconstruction Quality Issues
------------------------------

Noisy or Blurry Results
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Reconstructed focal spot or PSF looks noisy.

**Solutions:**

1. **Enable symmetrization**:

   .. code-block:: bash

       scopexr-fs --f "image.tif" --sym

2. **Change reconstruction filter**:

   .. code-block:: bash

       # Try smoother filters
       scopexr-fs --f "image.tif" --filter hamming
       # Options: ramp, shepp-logan, cosine, hamming, hann

3. **Increase profile averaging**:

   .. code-block:: yaml

       avg: true
       avg_number: 5  # Must be odd number

4. **Check input image quality** - noise in input propagates to output

Asymmetric Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Focal spot appears stretched or asymmetric.

**Solutions:**

1. **Use axis shift correction**:

   .. code-block:: yaml

       axis_shifts: 50  # Try larger values for more correction

2. **Verify magnification** is correct:

   .. code-block:: bash

       scopexr-fs --f "image.tif" --m 1.42  # Provide known value

3. **Check for centering errors** - re-run with better Hough parameters

4. **Verify test object is circular** - elliptical objects cause artifacts

Unrealistic Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** FWHM values seem too large or too small.

**Checklist:**

1. **Verify pixel size** (``--p`` parameter) matches your detector
2. **Check magnification** estimation or provide manually
3. **Confirm test object diameter** (``--d`` parameter) is correct
4. **Review image quality** - blurring affects measurements

Performance Issues
------------------

Analysis Takes Too Long
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Processing is very slow.

**Optimization tips:**

1. **Reduce number of angles**:

   .. code-block:: yaml

       nangles: 180  # Default is 360, try 180

2. **Decrease axis shift search**:

   .. code-block:: yaml

       axis_shifts: 20  # Reduce from 50

3. **Disable oversampling** for PSF:

   .. code-block:: bash

       scopexr-psf --f "image.tif" --no_oversample

4. **Check CPU usage** - close other applications

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Process crashes with memory error.

**Solutions:**

* Downscale large images before processing
* Reduce oversampling factor
* Close other applications to free RAM
* Process images one at a time in batch scripts

GUI Issues
----------

GUI Won't Start
~~~~~~~~~~~~~~~

**Symptom:** ``scopexr-gui`` command fails.

**Solutions:**

1. Verify PyQt6 is installed:

   .. code-block:: bash

       pip list | grep PyQt6

2. Test with CLI first to verify installation
3. Check for display/X11 issues on Linux
4. Run from command line to see error messages

Configuration Doesn't Load
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** GUI doesn't load YAML parameters.

**Solutions:**

1. Ensure ``fs_args.yaml`` and ``psf_args.yaml`` are in working directory
2. Check YAML syntax with online validator
3. Look for error messages in GUI console output
4. Create new config from examples folder

Output Files Issues
-------------------

No Output Files Generated
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Analysis completes but no files created.

**Checklist:**

1. Check output directory exists and is writable
2. Review console output for file I/O errors
3. Verify disk space is available
4. Check file permissions on output folder

Plots Look Wrong
~~~~~~~~~~~~~~~~

**Symptom:** Generated PNG files are blank or corrupted.

**Solutions:**

* Update matplotlib: ``pip install --upgrade matplotlib``
* Check for font-related warnings in output
* Try different matplotlib backend in code
* Verify image saving permissions

Error Messages
--------------

"IndexError: index out of bounds"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Likely causes:**

* Profile extraction failed (circle too small/large)
* Image dimensions don't match expected values
* Array indexing error in edge detection

**Solutions:**

* Adjust ``hl`` (half-length) parameter in config
* Verify circle detection found correct object
* Check image isn't truncated or corrupted

"ValueError: operands could not be broadcast"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Likely causes:**

* Dimension mismatch in sinogram reconstruction
* Incorrect image format or bit depth
* Parameter values incompatible with image size

**Solutions:**

* Verify all dimensions in config match image
* Review console output for dimension warnings

Getting Help
------------

If you continue experiencing issues:

1. **Review examples** - Compare your setup with working examples
2. **Simplify** - Try with provided sample images first
3. **Report bugs** - See :doc:`contributing` for how to file issues

**When reporting issues, include:**

* SCOPE-XR version (``pip show scopexr``)
* Python version (``python --version``)
* Operating system
* Full error message and traceback
* Minimal example to reproduce the issue
* Configuration file used (YAML)

For questions and support, open an issue on `GitHub <https://github.com/jacopoaltieri/scope-xr/issues>`_.
