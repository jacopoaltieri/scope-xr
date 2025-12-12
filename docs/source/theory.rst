Physical Methodology
====================

SCOPE-XR relies on established physical principles in X-ray optics to characterize system performance.

Focal Spot Characterization
---------------------------
The focal spot dimensions are evaluated based on the methodology proposed by **Di Domenico et al.** The fundamental relationship between the measured Full Width at Half Maximum (FWHM) and the physical focal spot size ($fs$) is given by:

.. math::

   fs = \frac{FWHM \cdot P}{M_{fs}}

Where:
   * :math:`P` is the pixel size (mm).
   * :math:`M_{fs}` is the magnification of the focal spot on the image plane.

The focal spot magnification is related to the total system magnification (:math:`M`) by:

.. math::

   M_{fs} = M - 1

If the user does not provide :math:`M`, the system estimates it automatically using the ratio of the physical test object radius to the detected radius on the image plane.

Detector PSF Analysis
---------------------
The Point Spread Function (PSF) reconstruction follows the approach by **Forster et al.** To achieve sub-pixel resolution, the software implements a circular sector oversampling strategy. This allows for a high-resolution reconstruction of the detector's response even when limited by the physical pixel pitch.

Mathematical Reconstruction
---------------------------
The core of the reconstruction engine is the **Filtered Back Projection (FBP)** algorithm.

1. **Sinogram Generation:** The derivative of radial profiles extracted from the circular edge creates a sinogram :math:`S(\theta, r)`.
2. **Filtering:** A choice of filters (Ramp, Shepp-Logan, Hamming, etc.) is applied to the sinogram to suppress 1/r blurring.
3. **Back Projection:** The filtered sinogram is back-projected to reconstruct the 2D distribution.