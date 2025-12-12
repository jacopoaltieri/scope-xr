Processing Pipeline
===================

SCOPE-XR follows a standardized 5-step pipeline for every analysis:

1. Input & Detection
--------------------
Images are loaded (PNG, TIF, or RAW). We use a **Hough Transform** to detect the circular aperture. A center-of-mass estimation refines the coordinates.

2. Geometric Validation
-----------------------
The program verifies the **straight-edge constraint** for focal spot analysis. If the sinogram is asymmetric due to centering errors, an automatic axis-shift correction is applied.

3. Profile Extraction
---------------------
Radial profiles are extracted from the center of the aperture outwards. The numerical derivative of these profiles forms the basis of the sinogram.

4. FBP Reconstruction
---------------------
Using the best-shifted sinogram, the Filtered Back Projection algorithm generates the 2D spatial representation of the source or the detector PSF.

5. Measurement & Fitting
------------------------
* **Focal Spot:** The FWHM of the reconstructed focal spot is measured along horizontal and vertical axes.
* **PSF:** A Gaussian curve is fitted to the sinogram profiles to determine horizontal and vertical FWHM.