Physical Methodology
====================

The full mathematical derivation of the penumbra analysis method is thoroughly presented by `Di Domenico et al. <https://aapm.onlinelibrary.wiley.com/doi/full/10.1118/1.4938414>`_ for focal spot (FS) reconstruction and adapted by `Forster et al. <https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-4-7409>`_ for detector Point-Spread Function (PSF) evaluation. This section summarizes the key theoretical concepts and equations implemented in the SCOPE-XR software.

The radiographic image on the detector can be modeled as the convolution of an ideal, geometrically projected image, with a 2D blurring kernel (FS or PSF, depending on the test object position with respect to the x-ray tube), both rescaled properly by their respective magnifications. If the ideal object is a circular object with a sharp edge, the blurring kernel can be extracted from the measured image by analyzing the penumbra of the recorded image.
It can be shown that, for any arbitrary edge orientation, the derivative of the intensity profile perpendicular to the edge is the Radon transform of the 2D blurring kernel along the direction of that profile. Since the circular edge spans all angles, the derivative of the radial profiles acquired at each angle can be arranged to form a sinogram, which is then inverted using standard tomographic reconstruction algorithms (e.g., Filtered Back Projection) to obtain the 2D blurring kernel.

Focal Spot Analysis
-------------------

For the FS analysis, the test object is placed close to the x-ray source, so that the blurring kernel corresponds to the focal spot distribution, magnified on the image plane. The focal spot size (:math:`fs`) is then computed from the Full Width at 15% Maximum (FW15M) of the reconstructed 2D distribution using the relation:

.. math::

   fs = \frac{FW15M \cdot P}{M_{fs}}

Where:
   * :math:`P` is the pixel size (mm).
   * :math:`M_{fs}` is the magnification of the focal spot on the image plane.

The focal spot magnification is related to the total system magnification (:math:`M`) by:

.. math::

   M_{fs} = M - 1

If the user does not provide :math:`M`, the system estimates it automatically using the ratio of the physical test object radius to the detected radius on the image plane.


Detector PSF Analysis
---------------------

The Point Spread Function (PSF) reconstruction follows the approach by `Forster et al. <https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-4-7409>`_ To achieve sub-pixel resolution, the software implements a circular sector oversampling strategy. This allows for a high-resolution reconstruction of the detector's response even when limited by the physical pixel pitch.
In this case, no magnification correction is needed, as the test object is placed in contact with the detector. The PSF FWHM is computed directly from the reconstructed 2D distribution in pixel units.

Oversampling
~~~~~~~~~~~~

The radial density of the reconstruction is ultimately limited by the size of the detector pixels. While this is not an issue for FS reconstruction, since typically its dimensions are much larger than the pixel size, the PSF of modern imaging systems often approach single-pixel widths. For this reason, the information on the dimensions and 2D distribution of the PSF is partially lost due to the limited amount of available sampling points.

To overcome this, the standard *slanted-edge method* is used, where a physical tilt of an edge relative to the pixel grid creates a composite, oversampled Edge Spread Function (ESF). Forster et al. adapted this concept, recognizing that the inherent curvature of the circular aperture provides these same sub-pixel shifts relative to the detector grid, thus enabling an oversampled reconstruction from the single acquired image.

SCOPE-XR implements this oversampling strategy, leaving to the user the choice of the angular aperture, the strength of the Gaussian blurring kernel, as well as the oversampling factors. However, to ensure the validity of the straight-edge approximation used in the derivation of the method, a new constraint on the maximum angular aperture for oversampling is introduced here and is automatically suggested by the software.

Let us define the sagitta :math:`\Delta r` as the deviation between the curved circular edge and its straight-line projection within this angular segment:

To ensure that this curvature remains unresolved by the oversampled imaging system, :math:`\Delta r` must not exceed the new effective pixel pitch:

.. math::

   \Delta r \leq \frac{P}{N}

Where :math:`P` is the pixel size and :math:`N` is the oversampling step (e.g., :math:`N=4` for 4x oversampling). By simple geometrical considerations, the value of the sagitta is :math:`\Delta r = R(1-\cos\alpha)`. By substituting this value into the pixel pitch constraint and inverting the equation, the constraint on the angle becomes:

.. math::

   \alpha \leq \arccos\left(1-\frac{P}{NR}\right)

This equation provides the maximum allowable angular aperture for a given :math:`R`, :math:`P`, and :math:`N` that robustly satisfies the straight-edge approximation for oversampling.

Since this relation shows that the allowable angular aperture decreases with increasing radius :math:`R`, it is useful to analyze the transverse extent :math:`\Delta y = R \tan\alpha`, which is proportional to the number of independent data points available for interpolation. In the limit of large :math:`R`, the arccosine in the angular aperture equation can be Taylor expanded at first order, obtaining :math:`\alpha_{\lim} \approx \sqrt{\frac{2P}{NR}}`. Substituting this into the transverse extent equation, we obtain:

.. math::

   \Delta y = R\tan\alpha_{\lim}\approx R \sqrt{\frac{2P}{NR}} = \sqrt{\frac{2PR}{N}}

Since :math:`\Delta y` increases with :math:`\sqrt{R}`, the optimal strategy is to select the largest possible circle radius that fits within the measurement field of view, as this maximizes the number of data points available for the oversampled profile.

Once :math:`R` is fixed, the oversampling should be performed using an angular aperture satisfying:

.. math::

   \delta \theta \leq 2\arccos\left(1-\frac{P}{NR}\right)

This constraint ensures robust sub-pixel resolution in the PSF reconstruction while maintaining the validity of the straight-edge approximation throughout the oversampling region.