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
from scipy.ndimage import shift
from skimage.transform import iradon

from .sinogram_extraction import symmetrize_sinogram


def reconstruct_focal_spot(
    sinogram: np.ndarray, filter_name: str, symmetrize: bool
) -> np.ndarray:
    """
    Reconstructs the focal spot image from sinogram via filtered back-projection.

    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles).
    filter_name
        Name of the filter to use in the inverse radon transform.
    symmetrize
        If True, average sinogram over 180 degrees before reconstruction.

    Returns
    -------
    np.ndarray
        2D array representing the reconstructed focal spot.
    """
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
    output_tiff_path: str,
    filter_name: str,
    shifts: list,
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

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    reconstructions = []
    # Prepare angles for full 360 degree sinogram
    n_angles = sinogram.shape[1]
    theta = np.linspace(0.0, 360.0, n_angles, endpoint=False)

    for delta in shifts:
        # shift sinogram vertically:
        #   shifting by +delta moves content down, so the effective axis moves up
        shifted_sino = shift(sinogram, shift=[delta, 0], order=3, mode="nearest")

        # reconstruct
        rec = iradon(shifted_sino, theta=theta, filter_name=filter_name)
        reconstructions.append(rec.astype(np.float32))

    tifffile.imwrite(
        output_tiff_path, np.stack(reconstructions, axis=0), photometric="minisblack"
    )
