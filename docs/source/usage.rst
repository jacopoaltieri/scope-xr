Usage
=====

SCOPE-XR provides two main interfaces designed for different user
workflows.

The program supports configurable execution via YAML configuration
files. You can find an example of these files in the ``examples``
folder, along with some simulated images to test the package.

Supported Image Formats
-----------------------

The supported input image formats are:

-  ``.png``
-  ``.tif``
-  ``.raw`` (must be accompanied by a corresponding ``.xml`` metadata
   file)

GUI Execution
-------------

The recommended way for routine analysis. It features live image
previews and interactive parameter tuning. To run the GUI, simply type:

.. code:: bash

   scopexr-gui

GUI Features:

-  **Easy Mode Selection**: Separate tabs for “Focal Spot (FS)” and
   “PSF” analysis.
-  **Automatic Configuration**: The GUI automatically loads all default
   parameters from ``fs_args.yaml`` or ``psf_args.yaml`` on startup.
-  **Image Preview**: Load any .png or .tif image to see a preview
   directly in the app.
-  **Full Parameter Control**: All CLI flags are editable via
   interactive widgets.
-  **Edit Config Files**: A button allows you to directly open and edit
   the default .yaml config file for the active tab.
-  **Live Output**: All console output from the analysis script is
   printed directly to a text box within the GUI.

CLI Execution
-------------

Ideal for batch processing and integration into automated research
pipelines. To run the program with the default settings (as defined in
``fs_args.yaml`` or ``psf_args.yaml``), use the following commands:

-  **Focal Spot:**

   .. code:: bash

      scopexr-fs --f "path/to/img.png"

-  **PSF:**

   .. code:: bash

      scopexr-psf --f "path/to/img.png"

Use the ``--help`` flag with any command to see all available
parameters.

Overriding Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can override any configuration value directly from the command line
by adding the corresponding flag. For example:

.. code:: bash

   scopexr-fs --f "path/to/img.png" --p 0.2

In this case, the pixel size will be set to ``0.2 mm`` instead of the
default value specified in the YAML file.

Available CLI Flags
~~~~~~~~~~~~~~~~~~~

Focal Spot CLI
^^^^^^^^^^^^^^

+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| **Flag**                | **Description**                                                                                                                               |
+=========================+===============================================================================================================================================+
| `--config` (str)        | Path to the YAML configuration file.                                                                                                          |
| `--f` (str, *required*) | Path to the input image file (`.raw`, `.png`, `.tif`).                                                                                        |
| `--o` (str)             | Output directory to store results.                                                                                                            |
| `--p` (float)           | Pixel size in mm.                                                                                                                             |
| `--d` (float)           | Physical diameter of the circular object in mm.                                                                                               |
| `--no_hough`            | Skip Hough Transform for automatic circle detection.                                                                                          |
| `--m` (float)           | Image magnification. If not provided, estimated automatically. Providing it from geometrical considerations may lead to more precise results. |
| `--n` (int)             | Minimum number of pixels required to achieve a reasonable focal spot size.                                                                    |
| `--nangles` (int)       | Number of angular projections for profile extraction.                                                                                         |
| `--hl` (int)            | Half length of the extracted radial profiles.                                                                                                 |
| `--ds` (int)            | Step size used for numerical derivative calculations.                                                                                         |
| `--axis_shifts` (int)   | Number of steps to shift the sinogram axis.                                                                                                   |
| `--filter` (str)        | Filter used during focal spot reconstruction. Options: `ramp`, `shepp-logan`, `cosine`, `hamming`, `hann`. Use `None` for no filter.          |
| `--avg_number` (int)    | Number of profiles to average, must be odd. Only used if `--avg` is true                                                                      |
| `--sym`                 | Symmetrize the sinogram before reconstruction.                                                                                                |
| `--shift`               | Enable automatic sinogram shifting.                                                                                                           |
| `--no_shift`            | Disable automatic sinogram shifting. (*Mutually exclusive with* `--shift`)                                                                    |
| `--avg`                 | Average neighboring sinogram profiles to improve FWHM estimation.                                                                             |
| `--no_avg`              | Do not average neighboring profiles. (*Mutually exclusive with* `--avg`)                                                                      |
| `--show`                | Display plots during processing (matplotlib windows).                                                                                         |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+


PSF CLI
^^^^^^^

+-------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Flag**                | **Description**                                                                                                                                                                                                                                                                                                                                                   |
+=========================+===================================================================================================================================================================================================================================================================================================================================================================+
| `--config` (str)        | Path to the YAML configuration file.                                                                                                                                                                                                                                                                                                                              |
| `--f` (str, *required*) | Path to the input image file (`.raw`, `.png`, `.tif`).                                                                                                                                                                                                                                                                                                            |
| `--o` (str)             | Output directory to store results.                                                                                                                                                                                                                                                                                                                                |
| `--p` (float)           | Pixel size in mm.                                                                                                                                                                                                                                                                                                                                                 |
| `--d` (float)           | Physical diameter of the circular object in mm.                                                                                                                                                                                                                                                                                                                   |
| `--no_hough`            | Skip Hough Transform for automatic circle detection.                                                                                                                                                                                                                                                                                                              |
| `--nangles` (int)       | Number of angular projections for profile extraction.                                                                                                                                                                                                                                                                                                             |
| `--hl` (int)            | Half length of the extracted radial profiles.                                                                                                                                                                                                                                                                                                                     |
| `--ds` (int)            | Step size used for numerical derivative calculations.                                                                                                                                                                                                                                                                                                             |
| `--axis_shifts` (int)   | Number of steps to shift the sinogram axis.                                                                                                                                                                                                                                                                                                                       |
| `--filter` (str)        | Filter used during focal spot reconstruction. Options: `ramp`, `shepp-logan`, `cosine`, `hamming`, `hann`. Use `None` for no filter.                                                                                                                                                                                                                              |
| `--avg_number` (int)    | Number of profiles to average, must be odd. Only used if `--avg` is true                                                                                                                                                                                                                                                                                          |
| `--sym`                 | Symmetrize the sinogram before reconstruction.                                                                                                                                                                                                                                                                                                                    |
| `--dtheta`              | Angle of the circular sector for oversampling (in degrees).                                                                                                                                                                                                                                                                                                       |
| `--resample1`           | First resample factor (fine grid), used only with 3-step oversampling.                                                                                                                                                                                                                                                                                            |
| `--resample2`           | Second resample factor (coarse grid). This will be the final oversampling factor.                                                                                                                                                                                                                                                                                 |
| `--gaussian_sigma`      | Standard deviation of the gaussian blur applied between the fine and the coarse resampling, used only with 3-step oversampling.                                                                                                                                                                                                                                   |
| `--oversample_strategy` | Choose oversampling strategy: `1` or `2`. Default is `1` when oversampling is enabled. `1` is the traditional oversampling method, `2` is the 3-step method proposed by [Forster et al.](https://www.researchgate.net/publication/387092230_Single-shot_2D_detector_point-spread_function_analysis_employing_a_circular_aperture_and_a_back-projection_approach). |
| `--shift`               | Enable automatic sinogram shifting.                                                                                                                                                                                                                                                                                                                               |
| `--no_shift`            | Disable automatic sinogram shifting. (*Mutually exclusive with* `--shift`)                                                                                                                                                                                                                                                                                        |
| `--avg`                 | Average neighboring sinogram profiles to improve FWHM estimation.                                                                                                                                                                                                                                                                                                 |
| `--no_avg`              | Do not average neighboring profiles. (*Mutually exclusive with* `--avg`)                                                                                                                                                                                                                                                                                          |
| `--oversample`          | Performs oversampling.                                                                                                                                                                                                                                                                                                                                            |
| `--no_oversample`       | Disables oversampling. (*Mutually exclusive with* `--oversample`)                                                                                                                                                                                                                                                                                                 |
| `--show`                | Display plots during processing (matplotlib windows).                                                                                                                                                                                                                                                                                                             |
+-------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
