# SCOPE-XR (Single-image Characterization Of PErformance in X-Ray systems)
# Copyright (C) 2026  Jacopo Altieri
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import shift as ndimage_shift
from skimage.transform import iradon
from scipy.ndimage import shift as ndimage_shift

from .sinogram_extraction import symmetrize_sinogram, manual_center_sinogram


def _center_sinogram_columns(sinogram: np.ndarray) -> np.ndarray:
    """
    Aligns each angular column's center of mass to the radial midpoint of the sinogram.
    This is only used when reconstructing the detector PSF.
    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles).

    Returns
    -------
    np.ndarray
        2D array of shape (n_rays, n_angles) with each column shifted so its
        center of mass lies at the central radial index.

    Raises
    ------
    ValueError
        If ``sinogram`` is not a two-dimensional array.
    """
    sinogram = np.asarray(sinogram)
    if sinogram.ndim != 2:
        raise ValueError("sinogram must be a two-dimensional array")

    n_rays, n_angles = sinogram.shape
    out_dtype = (
        sinogram.dtype if np.issubdtype(sinogram.dtype, np.floating) else np.float64
    )
    if n_rays < 2 or n_angles == 0:
        return sinogram.astype(out_dtype, copy=True)

    centered = np.empty((n_rays, n_angles), dtype=np.float64)
    center_idx = (n_rays - 1) / 2.0
    ray_indices = np.arange(n_rays, dtype=np.float64)

    for j in range(n_angles):
        col = sinogram[:, j].astype(np.float64)
        finite_mask = np.isfinite(col)
        if not np.any(finite_mask):
            centered[:, j] = col
            continue

        peak = np.max(col[finite_mask])
        if peak <= 0.0:
            centered[:, j] = col
            continue

        # Suppress baseline tail noise (< 10% of peak) so far-field noise cannot bias CoM
        weights = np.where(finite_mask, np.maximum(col - 0.1 * peak, 0.0), 0.0)
        w_sum = weights.sum()
        if w_sum <= 0.0:
            centered[:, j] = col
            continue

        com = np.dot(ray_indices, weights) / w_sum
        delta = center_idx - com
        centered[:, j] = ndimage_shift(col, shift=delta, order=1, mode="nearest")

    return centered.astype(out_dtype, copy=False)


def reconstruct_blurring_kernel(
    sinogram: np.ndarray,
    filter_name: str,
    symmetrize: bool,
    center_columns: bool = False,
) -> np.ndarray:
    """
    Reconstructs the blurring kernel image from sinogram via filtered back-projection.

    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles).
    filter_name
        Name of the filter to use in the inverse radon transform.
    symmetrize
        If True, average sinogram over 180 degrees before reconstruction.
    center_columns
        If True, aligns each angular column's center of mass to the radial
        midpoint before symmetrization and back-projection. Recommended for
        detector PSF reconstruction (`True`), and disabled by default (`False`)
        for focal spot (FS) reconstruction.

    Returns
    -------
    np.ndarray
        2D array representing the reconstructed blurring kernel.
    """
    if center_columns:
        sinogram = _center_sinogram_columns(sinogram)

    if symmetrize:
        sinogram = symmetrize_sinogram(sinogram)
        theta = np.linspace(0.0, 180.0, sinogram.shape[1], endpoint=False)
        reconstruction = iradon(
            sinogram, theta=theta, filter_name=filter_name, circle=True
        )
    else:
        theta = np.linspace(0.0, 360.0, sinogram.shape[1], endpoint=False)
        reconstruction = iradon(
            sinogram, theta=theta, filter_name=filter_name, circle=True
        )
    return reconstruction


def reconstruct_with_axis_shifts(
    sinogram: np.ndarray,
    output_tiff_path: str | Path,
    filter_name: str,
    shifts: list,
    symmetrize: bool = False,
    center_columns: bool = False,
) -> None:
    """
    Applies multiple vertical shifts to a sinogram, reconstructs each, and saves as a multi-page TIFF.

    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles).
    output_tiff_path
        Path for the output multi-page TIFF file.
    filter_name
        Filter name for the inverse radon transform.
    shifts
        List of integer shifts (rows) to apply to sinogram.
    symmetrize
        If True, reconstruction is performed with 180-degree sinogram symmetrization.
    center_columns
        If True, aligns each angular column's center of mass to the radial
        midpoint prior to applying the axis shifts. Defaults to False.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    reconstructions = []

    max_abs_shift = max(abs(int(s)) for s in shifts) if shifts else 0
    target_rows = sinogram.shape[0] - max_abs_shift
    if target_rows <= 0:
        raise ValueError("Axis shifts are too large for the sinogram height.")

    if center_columns:
        sinogram = _center_sinogram_columns(sinogram)

    for delta in shifts:
        shifted_sino, _ = manual_center_sinogram(sinogram, delta)
        if shifted_sino.shape[0] > target_rows:
            trim = shifted_sino.shape[0] - target_rows
            top = trim // 2
            shifted_sino = shifted_sino[top : top + target_rows, :]
        rec = reconstruct_blurring_kernel(
            shifted_sino, filter_name, symmetrize, center_columns=False
        )
        reconstructions.append(rec.astype(np.float32))

    tifffile.imwrite(
        output_tiff_path, np.stack(reconstructions, axis=0), photometric="minisblack"
    )