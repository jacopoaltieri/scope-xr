Physical Methodology
====================

The full mathematical derivation of the penumbra analysis method is thoroughly presented by `Di Domenico et al. <https://aapm.onlinelibrary.wiley.com/doi/full/10.1118/1.4938414>`_ for focal spot (FS) reconstruction and adapted by `Forster et al. <https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-4-7409>`_ for detector Point-Spread Function (PSF) evaluation. This section summarizes the key theoretical concepts and equations implemented in the SCOPE-XR software.

Image Formation and Mathematical Principles
-------------------------------------------

In a linear, shift-invariant imaging system, a radiographic image
:math:`I(x,y)` recorded on the detector plane is modeled as the 2D
convolution of an ideal, geometrically projected object transmittance
:math:`A(x,y)` with a blurring kernel :math:`K(x,y)`:

.. math::

   I(x,y) = A(x,y) \ast K(x,y)

Depending on the test object position relative to the X-ray source and
detector, :math:`K(x,y)` corresponds to either the magnified focal
spot emission distribution or the detector PSF.

For a sharp circular absorber, the derivative of the intensity profile
perpendicular to the edge at orientation :math:`\theta` yields the
Line Spread Function (LSF). Mathematically, this radial derivative
represents the Radon transform :math:`g(r, \theta)` of the 2D blurring
kernel along that orientation:

.. math::

   g(r, \theta) =
   \frac{\partial I_r(r, \theta)}{\partial r} =
   \int_{-\infty}^{+\infty}
   \int_{-\infty}^{+\infty}
   \delta(r - x\cos\theta - y\sin\theta)
   K(x,y)\,dx\,dy

Differentiating radial profiles extracted across :math:`360^\circ`
forms a sinogram, which is centered/aligned and inverted via Filtered
Back-Projection (FBP) or iterative reconstruction algorithms (e.g.,
EM-ML) to recover the 2D distribution.

Focal Spot Analysis
-------------------

For focal spot (FS) analysis, the test object is placed near the X-ray
source at source-to-object distance :math:`p` and object-to-detector
distance :math:`q`, resulting in a total system magnification
:math:`M = (p+q)/p`. The focal spot magnification on the image plane
is:

.. math::

   M\_{fs} = M - 1

If :math:`M` is not provided by the user, SCOPE-XR estimates it
automatically using the ratio of the physical absorber radius to the
detected radius on the image plane.

The physical focal spot size :math:`fs` is calculated from the Full
Width at 15% Maximum (FW15M)—in accordance with IEC 60336 standards—or
Full Width at Half Maximum (FWHM) using the relation:

.. math::

   fs = \frac{\text{FW15M} \cdot P}{M_{fs}}

where :math:`P` is the detector pixel pitch (mm).

Straight-Edge Approximation Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure each radial segment functions locally as a straight edge, the
curvature displacement (sagitta) :math:`\Delta` across the projected
focal spot extent :math:`a_* = (M-1)a` must not exceed the detector
pixel pitch :math:`P`:

.. math::

   \Delta = M R -
   \sqrt{M^2 R^2 - \frac{(M-1)^2 a^2}{4}} \le P

This condition imposes lower bounds on the absorber radius
:math:`R \ge \frac{(1+n^2)P}{2M}` (where :math:`n \ge 6` represents
the minimal required penumbra pixel span) and system magnification
:math:`M > \frac{a + n P}{a}`.

Detector PSF Analysis
---------------------

For detector PSF analysis, the circular absorber is placed in direct
contact with the detector (:math:`M=1, M_{fs}=0`), rendering focal
spot penumbra magnification negligible (:math:`a_* \approx 0`).
Quantitative metrics—such as the Gaussian-fitted FWHM and the 1D
Modulation Transfer Function (MTF) derived via Fourier transform—are
evaluated directly in pixel units.

Oversampling
~~~~~~~~~~~~

Because digital detector pixel pitch limits radial sampling density,
native sampling causes shape loss and algorithmic broadening for narrow
PSFs approaching single-pixel widths. Adapted from the slanted-edge
method, circular edge curvature provides inherent sub-pixel shifts
relative to the detector grid.

SCOPE-XR implements two oversampling strategies:

1. **Three-step pipeline (Forster et al.):** Groups pixels within an
   angular sector :math:`\delta\theta`, interpolates them onto a fine
   grid, applies a Gaussian blur :math:`G_{\sigma_{\text{smooth}}}` to
   suppress noise and angular cross-talk, and resamples onto the target
   sub-pixel grid prior to differentiation.
2. **Direct single-step pipeline (recommended):** Interpolates pixels directly onto
   the target sub-pixel grid without Gaussian blur, avoiding additional
   algorithmic broadening when strict isolation of the intrinsic PSF is
   needed.

Oversampling Angular Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To maintain the validity of the straight-edge approximation within an
oversampled pixel pitch :math:`P/N` (where :math:`N` is the
oversampling factor), the edge sagitta
:math:`\Delta r = R(1-\cos\alpha)` must not exceed :math:`P/N`:

.. math::

   \Delta r \le \frac{P}{N}

Solving for :math:`\alpha` yields the maximum allowable half-angle:

.. math::

   \alpha \le \arccos\left(1
   - \frac{P}{N R}\right)

The maximum allowable angular aperture for an oversampling sector is
therefore:

.. math::

   \delta\theta \le 2\arccos\left(1
   - \frac{P}{N R}\right)

A first-order Taylor expansion at large :math:`R` yields
:math:`\alpha_{\text{lim}} \approx \sqrt{\frac{2P}{NR}}`, giving a
transverse sampling extent:

.. math::

   \Delta y = R
   \tan\alpha\_{\text{lim}}
   \approx \sqrt{\frac{2 P R}{N}}

Because :math:`\Delta y` grows with :math:`\sqrt{R}`, selecting the
largest circular aperture that fits within the detector field of view
maximizes the number of independent data points available for
oversampled profile reconstruction. For general PSF evaluation without
sub-pixel oversampling, curvature displacements up to
:math:`\Delta \le 1\text{--}2\text{ px}` preserve high reconstruction
fidelity without introducing visible distortion artifacts.
