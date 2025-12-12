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

+---------+------------------------------------------------------------+
| *       | **Description**                                            |
| *Flag** |                                                            |
+=========+============================================================+
| ``--c   | Path to the YAML configuration file.                       |
| onfig`` |                                                            |
| (str)   |                                                            |
+---------+------------------------------------------------------------+
| ``--f`` | Path to the input image file (``.raw``, ``.png``,          |
| (str,   | ``.tif``).                                                 |
| *req    |                                                            |
| uired*) |                                                            |
+---------+------------------------------------------------------------+
| ``--o`` | Output directory to store results.                         |
| (str)   |                                                            |
+---------+------------------------------------------------------------+
| ``--p`` | Pixel size in mm.                                          |
| (float) |                                                            |
+---------+------------------------------------------------------------+
| ``--d`` | Physical diameter of the circular object in mm.            |
| (float) |                                                            |
+---------+------------------------------------------------------------+
| ``--no_ | Skip Hough Transform for automatic circle detection.       |
| hough`` |                                                            |
+---------+------------------------------------------------------------+
| ``--m`` | Image magnification. If not provided, estimated            |
| (float) | automatically. Providing it from geometrical               |
|         | considerations may lead to more precise results.           |
+---------+------------------------------------------------------------+
| ``--n`` | Minimum number of pixels required to achieve a reasonable  |
| (int)   | focal spot size.                                           |
+---------+------------------------------------------------------------+
| ``--na  | Number of angular projections for profile extraction.      |
| ngles`` |                                                            |
| (int)   |                                                            |
+---------+------------------------------------------------------------+
| `       | Half length of the extracted radial profiles.              |
| `--hl`` |                                                            |
| (int)   |                                                            |
+---------+------------------------------------------------------------+
| `       | Step size used for numerical derivative calculations.      |
| `--ds`` |                                                            |
| (int)   |                                                            |
+---------+------------------------------------------------------------+
| ``-     | Number of steps to shift the sinogram axis.                |
| -axis_s |                                                            |
| hifts`` |                                                            |
| (int)   |                                                            |
+---------+------------------------------------------------------------+
| ``--f   | Filter used during focal spot reconstruction. Options:     |
| ilter`` | ``ramp``, ``shepp-logan``, ``cosine``, ``hamming``,        |
| (str)   | ``hann``. Use ``None`` for no filter.                      |
+---------+------------------------------------------------------------+
| ``      | Number of profiles to average, must be odd. Only used if   |
| --avg_n | ``--avg`` is true                                          |
| umber`` |                                                            |
| (int)   |                                                            |
+---------+------------------------------------------------------------+
| ``      | Symmetrize the sinogram before reconstruction.             |
| --sym`` |                                                            |
+---------+------------------------------------------------------------+
| ``--    | Enable automatic sinogram shifting.                        |
| shift`` |                                                            |
+---------+------------------------------------------------------------+
| ``--no_ | Disable automatic sinogram shifting. (*Mutually exclusive  |
| shift`` | with* ``--shift``)                                         |
+---------+------------------------------------------------------------+
| ``      | Average neighboring sinogram profiles to improve FWHM      |
| --avg`` | estimation.                                                |
+---------+------------------------------------------------------------+
| ``--n   | Do not average neighboring profiles. (*Mutually exclusive  |
| o_avg`` | with* ``--avg``)                                           |
+---------+------------------------------------------------------------+
| ``-     | Display plots during processing (matplotlib windows).      |
| -show`` |                                                            |
+---------+------------------------------------------------------------+

PSF CLI
^^^^^^^

+---+------------------------------------------------------------------+
| * | **Description**                                                  |
| * |                                                                  |
| F |                                                                  |
| l |                                                                  |
| a |                                                                  |
| g |                                                                  |
| * |                                                                  |
| * |                                                                  |
+===+==================================================================+
| ` | Path to the YAML configuration file.                             |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| c |                                                                  |
| o |                                                                  |
| n |                                                                  |
| f |                                                                  |
| i |                                                                  |
| g |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| s |                                                                  |
| t |                                                                  |
| r |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Path to the input image file (``.raw``, ``.png``, ``.tif``).     |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| f |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| s |                                                                  |
| t |                                                                  |
| r |                                                                  |
| , |                                                                  |
| * |                                                                  |
| r |                                                                  |
| e |                                                                  |
| q |                                                                  |
| u |                                                                  |
| i |                                                                  |
| r |                                                                  |
| e |                                                                  |
| d |                                                                  |
| * |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Output directory to store results.                               |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| o |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| s |                                                                  |
| t |                                                                  |
| r |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Pixel size in mm.                                                |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| p |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| f |                                                                  |
| l |                                                                  |
| o |                                                                  |
| a |                                                                  |
| t |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Physical diameter of the circular object in mm.                  |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| d |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| f |                                                                  |
| l |                                                                  |
| o |                                                                  |
| a |                                                                  |
| t |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Skip Hough Transform for automatic circle detection.             |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| n |                                                                  |
| o |                                                                  |
| _ |                                                                  |
| h |                                                                  |
| o |                                                                  |
| u |                                                                  |
| g |                                                                  |
| h |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Number of angular projections for profile extraction.            |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| n |                                                                  |
| a |                                                                  |
| n |                                                                  |
| g |                                                                  |
| l |                                                                  |
| e |                                                                  |
| s |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| i |                                                                  |
| n |                                                                  |
| t |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Half length of the extracted radial profiles.                    |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| h |                                                                  |
| l |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| i |                                                                  |
| n |                                                                  |
| t |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Step size used for numerical derivative calculations.            |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| d |                                                                  |
| s |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| i |                                                                  |
| n |                                                                  |
| t |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Number of steps to shift the sinogram axis.                      |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| a |                                                                  |
| x |                                                                  |
| i |                                                                  |
| s |                                                                  |
| _ |                                                                  |
| s |                                                                  |
| h |                                                                  |
| i |                                                                  |
| f |                                                                  |
| t |                                                                  |
| s |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| i |                                                                  |
| n |                                                                  |
| t |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Filter used during focal spot reconstruction. Options: ``ramp``, |
| ` | ``shepp-logan``, ``cosine``, ``hamming``, ``hann``. Use ``None`` |
| - | for no filter.                                                   |
| - |                                                                  |
| f |                                                                  |
| i |                                                                  |
| l |                                                                  |
| t |                                                                  |
| e |                                                                  |
| r |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| s |                                                                  |
| t |                                                                  |
| r |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Number of profiles to average, must be odd. Only used if         |
| ` | ``--avg`` is true                                                |
| - |                                                                  |
| - |                                                                  |
| a |                                                                  |
| v |                                                                  |
| g |                                                                  |
| _ |                                                                  |
| n |                                                                  |
| u |                                                                  |
| m |                                                                  |
| b |                                                                  |
| e |                                                                  |
| r |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
| ( |                                                                  |
| i |                                                                  |
| n |                                                                  |
| t |                                                                  |
| ) |                                                                  |
+---+------------------------------------------------------------------+
| ` | Symmetrize the sinogram before reconstruction.                   |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| s |                                                                  |
| y |                                                                  |
| m |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Angle of the circular sector for oversampling (in degrees).      |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| d |                                                                  |
| t |                                                                  |
| h |                                                                  |
| e |                                                                  |
| t |                                                                  |
| a |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | First resample factor (fine grid), used only with 3-step         |
| ` | oversampling.                                                    |
| - |                                                                  |
| - |                                                                  |
| r |                                                                  |
| e |                                                                  |
| s |                                                                  |
| a |                                                                  |
| m |                                                                  |
| p |                                                                  |
| l |                                                                  |
| e |                                                                  |
| 1 |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Second resample factor (coarse grid). This will be the final     |
| ` | oversampling factor.                                             |
| - |                                                                  |
| - |                                                                  |
| r |                                                                  |
| e |                                                                  |
| s |                                                                  |
| a |                                                                  |
| m |                                                                  |
| p |                                                                  |
| l |                                                                  |
| e |                                                                  |
| 2 |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Standard deviation of the gaussian blur applied between the fine |
| ` | and the coarse resampling, used only with 3-step oversampling.   |
| - |                                                                  |
| - |                                                                  |
| g |                                                                  |
| a |                                                                  |
| u |                                                                  |
| s |                                                                  |
| s |                                                                  |
| i |                                                                  |
| a |                                                                  |
| n |                                                                  |
| _ |                                                                  |
| s |                                                                  |
| i |                                                                  |
| g |                                                                  |
| m |                                                                  |
| a |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Choose oversampling strategy: ``1`` or ``2``. Default is ``1``   |
| ` | when oversampling is enabled. ``1`` is the traditional           |
| - | oversampling method, ``2`` is the 3-step method proposed by      |
| - | `Forster et                                                      |
| o | al. <https://www.researchgate.net/publication/38                 |
| v | 7092230_Single-shot_2D_detector_point-spread_function_analysis_e |
| e | mploying_a_circular_aperture_and_a_back-projection_approach>`__. |
| r |                                                                  |
| s |                                                                  |
| a |                                                                  |
| m |                                                                  |
| p |                                                                  |
| l |                                                                  |
| e |                                                                  |
| _ |                                                                  |
| s |                                                                  |
| t |                                                                  |
| r |                                                                  |
| a |                                                                  |
| t |                                                                  |
| e |                                                                  |
| g |                                                                  |
| y |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Enable automatic sinogram shifting.                              |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| s |                                                                  |
| h |                                                                  |
| i |                                                                  |
| f |                                                                  |
| t |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Disable automatic sinogram shifting. (*Mutually exclusive with*  |
| ` | ``--shift``)                                                     |
| - |                                                                  |
| - |                                                                  |
| n |                                                                  |
| o |                                                                  |
| _ |                                                                  |
| s |                                                                  |
| h |                                                                  |
| i |                                                                  |
| f |                                                                  |
| t |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Average neighboring sinogram profiles to improve FWHM            |
| ` | estimation.                                                      |
| - |                                                                  |
| - |                                                                  |
| a |                                                                  |
| v |                                                                  |
| g |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Do not average neighboring profiles. (*Mutually exclusive with*  |
| ` | ``--avg``)                                                       |
| - |                                                                  |
| - |                                                                  |
| n |                                                                  |
| o |                                                                  |
| _ |                                                                  |
| a |                                                                  |
| v |                                                                  |
| g |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Performs oversampling.                                           |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| o |                                                                  |
| v |                                                                  |
| e |                                                                  |
| r |                                                                  |
| s |                                                                  |
| a |                                                                  |
| m |                                                                  |
| p |                                                                  |
| l |                                                                  |
| e |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Disables oversampling. (*Mutually exclusive with*                |
| ` | ``--oversample``)                                                |
| - |                                                                  |
| - |                                                                  |
| n |                                                                  |
| o |                                                                  |
| _ |                                                                  |
| o |                                                                  |
| v |                                                                  |
| e |                                                                  |
| r |                                                                  |
| s |                                                                  |
| a |                                                                  |
| m |                                                                  |
| p |                                                                  |
| l |                                                                  |
| e |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
| ` | Display plots during processing (matplotlib windows).            |
| ` |                                                                  |
| - |                                                                  |
| - |                                                                  |
| s |                                                                  |
| h |                                                                  |
| o |                                                                  |
| w |                                                                  |
| ` |                                                                  |
| ` |                                                                  |
+---+------------------------------------------------------------------+
