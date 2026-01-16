# SCOPE-XR (Single-image Characterization Of PErformance in X-Ray systems)
# Copyright (C) 2025  Jacopo Altieri
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
from scipy.ndimage import map_coordinates, shift, gaussian_filter1d
from scipy.stats import binned_statistic
from skimage.transform import iradon


def _check_phl(
    img: np.ndarray, cx: float, cy: float, radius: float, profile_half_length: int
) -> int:
    """
    Adjusts profile_half_length to avoid crossing *any* image boundary.

    Parameters
    ----------
    img
        2D grayscale image array.
    cx
        X-coordinate of the circle center.
    cy
        Y-coordinate of the circle center.
    radius
        Radius of the circle.
    profile_half_length
        Desired half-length of the sampling profile.

    Returns
    -------
    int
        Adjusted profile_half_length that fits within image bounds.

    Raises
    ------
    ValueError
        If the circle is too close to the edge to sample any profile.
    """
    img_h, img_w = img.shape

    # Find the shortest distance from the center (cx, cy) to any of the 4 edges.
    dist_to_left = cx
    dist_to_right = img_w - 1 - cx
    dist_to_top = cy
    dist_to_bottom = img_h - 1 - cy

    # The minimum of these is the largest radius we can *ever* sample
    # from the center without going out of bounds.
    max_sample_radius = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

    # The outermost point we sample is at (radius + profile_half_length).
    # This must be less than our maximum allowed sample radius.

    # Calculate the maximum possible profile_half_length
    # We subtract 1 as a safety margin (to avoid being exactly on the edge pixel)
    max_allowed_phl = int(max_sample_radius - radius - 1)

    if max_allowed_phl <= 0:
        # This means the circle radius itself is larger than the distance to an edge
        raise ValueError(
            f"Circle (radius={radius}) is too close to the edge. "
            f"Max allowed sample radius from center is {max_sample_radius:.2f} px. "
            f"Crop your image with more margin."
        )

    if profile_half_length > max_allowed_phl:
        print(
            f"Warning: profile_half_length reduced from {profile_half_length} "
            f"to {max_allowed_phl} to avoid crossing image border."
        )
        return max_allowed_phl

    # If the requested phl is already fine, return it
    return profile_half_length


def compute_profiles_and_sinogram(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int,
    profile_half_length: int,
    derivative_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts radial edge profiles around a circle and computes the sinogram via derivative.

    Parameters
    ----------
    img
        2D grayscale image array.
    cx
        X-coordinate of the circle center.
    cy
        Y-coordinate of the circle center.
    radius
        Radius of the circle.
    n_angles
        Number of angular samples around the circle.
    profile_half_length
        Half-length (in pixels) of the radial sampling profile.
    derivative_step
        Step size for computing the radial derivative.

    Returns
    -------
    profiles : np.ndarray
        2D array of shape (profile_length, n_angles), radial profiles.
    sinogram : np.ndarray
        2D array of shape (profile_length, n_angles), negative radial derivative profiles.
    """
    profile_half_length = _check_phl(img, cx, cy, radius, profile_half_length)

    # Sample angles in COUNTER-CLOCKWISE order to match iradon's convention.
    # Note: due to image coordinate system (y increases downward), we negate the angle
    angles = -np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    profile_length = int(2 * profile_half_length)

    # Pre-allocate output arrays
    profiles = np.zeros((n_angles, profile_length), dtype=np.float32)

    # Pre-compute radial coordinates once
    d_coords = np.arange(profile_length, dtype=np.float32) - profile_half_length

    # Pre-compute all trigonometric values (vectorized)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Vectorized computation of all sampling coordinates
    # Shape: (n_angles, profile_length)
    px = cx + np.outer(cos_angles, (radius + d_coords))
    py = cy + np.outer(sin_angles, (radius + d_coords))

    # Sample all profiles at once using vectorized map_coordinates
    for i in range(n_angles):
        profiles[i, :] = map_coordinates(
            img, [py[i], px[i]], order=1, mode="constant", cval=0.0
        )

    # Compute the derivative to obtain the sinogram
    sinogram = np.gradient(profiles, derivative_step, axis=1)
    return profiles.T, -sinogram.T


def _compute_polar_coordinates(
    cx: float,
    cy: float,
    img_shape: tuple[int, int],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes polar coordinates (angle and radial distance) for all pixels relative to circle center.

    Parameters
    ----------
    cx
        X-coordinate of the circle center.
    cy
        Y-coordinate of the circle center.
    img_shape
        Shape of the image (height, width).
    radius
        Radius of the circle.

    Returns
    -------
    phis : np.ndarray
        2D array of angular positions (radians) for each pixel.
    rs : np.ndarray
        2D array of radial distances from the circle edge for each pixel.
    """
    # Coordinates relative to center
    ys, xs = np.indices(img_shape)
    xs = xs.astype(np.float32) - cx
    ys = ys.astype(np.float32) - cy
    phis = np.arctan2(ys, xs)
    rs = np.hypot(xs, ys) - radius
    return phis, rs


def _extract_wedge_radial_samples(
    phis: np.ndarray,
    rs: np.ndarray,
    img: np.ndarray,
    theta: float,
    half_wedge: float,
    min_r: float,
    max_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts and sorts pixel samples from an angular wedge within a radial range.

    Parameters
    ----------
    phis
        2D array of angular positions (radians) for each pixel.
    rs
        2D array of radial distances from circle edge for each pixel.
    img
        2D grayscale image array.
    theta
        Central angle (radians) of the wedge.
    half_wedge
        Half-width (radians) of the angular wedge.
    min_r
        Minimum radial distance to include.
    max_r
        Maximum radial distance to include.

    Returns
    -------
    r_vals : np.ndarray
        Sorted radial distances of pixels in wedge.
    intensities : np.ndarray
        Corresponding pixel intensities, sorted by radial distance.
    """
    # Mask pixels in angular wedge
    dphi = (phis - theta + np.pi) % (2 * np.pi) - np.pi
    mask = np.abs(dphi) <= half_wedge
    r_vals = rs[mask]
    intensities = img[mask]

    # Restrict to radial range
    radial_mask = (r_vals >= min_r) & (r_vals <= max_r)
    r_vals = r_vals[radial_mask]
    intensities = intensities[radial_mask]

    # Sort by radial distance
    if r_vals.size > 0:
        idx = np.argsort(r_vals)
        r_vals = r_vals[idx]
        intensities = intensities[idx]

    return r_vals, intensities


def compute_subpixel_profiles_and_sinogram(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int,
    profile_half_length: int,
    derivative_step: int,
    dtheta: float,
    resample: float,
    gaussian_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes sub-pixel edge profiles and sinogram via oversampled binning in angular wedges.

    This is a unified function that handles both traditional and Gaussian-smoothed approaches:
    - If gaussian_sigma=0: Uses direct binned statistics with interpolation (traditional approach)
    - If gaussian_sigma>0: Uses fine grid → Gaussian blur → coarse grid pipeline (3-step approach)

    For more information about the 3-step method, see:
    https://www.researchgate.net/publication/387092230_Single-shot_2D_detector_point-spread_function_analysis_employing_a_circular_aperture_and_a_back-projection_approach

    Parameters
    ----------
    img
        2D grayscale image array.
    cx
        X-coordinate of the circle center.
    cy
        Y-coordinate of the circle center.
    radius
        Radius of the circle.
    n_angles
        Number of angular samples (in full 360°).
    profile_half_length
        Half-length (in pixels) of radial sampling.
    derivative_step
        Step size for derivative computation.
    dtheta
        Angular width (degrees) of wedge around each angle.
    resample
        Radial step for the final sampling grid (in pixels).
        When gaussian_sigma=0, this is the only resampling factor.
        When gaussian_sigma>0, this acts as resample2 (coarse grid) and fine grid is resample*100.
    gaussian_sigma
        Sigma for Gaussian smoothing (in units of fine grid spacing).
        If 0.0 (default): Uses traditional binned statistics approach.
        If > 0.0: Uses 3-step oversampled approach with Gaussian smoothing.

    Returns
    -------
    profiles : np.ndarray
        2D array of shape (profile_bins, n_angles), radial profiles.
    sinogram : np.ndarray
        2D array of shape (profile_bins, n_angles), negative radial derivatives.
    """
    profile_half_length = _check_phl(img, cx, cy, radius, profile_half_length)

    # Convert angles and angular wedge width to radians
    # Sample angles in COUNTER-CLOCKWISE order to match iradon's convention.
    # Note: due to image coordinate system (y increases downward), we negate the angle
    angles = -np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    half_wedge = np.deg2rad(dtheta) / 2

    # Coordinates relative to center
    phis, rs = _compute_polar_coordinates(cx, cy, img.shape, radius)

    # Set up radial grid for interpolation
    min_r = -profile_half_length
    max_r = profile_half_length

    if gaussian_sigma == 0.0:
        # TRADITIONAL APPROACH: Direct binned statistics with interpolation
        r_grid = np.arange(min_r, max_r + 1 / resample, 1 / resample)
        n_bins = r_grid.size
        profiles = np.full((n_angles, n_bins), np.nan, dtype=np.float32)

        for i, theta in enumerate(angles):
            r_vals, intensities = _extract_wedge_radial_samples(
                phis, rs, img, theta, half_wedge, min_r, max_r
            )
            bin_edges = np.append(r_grid, r_grid[-1] + (1 / resample))

            # Interpolate to uniform grid
            # Handle empty wedge case to avoid interp error
            if r_vals.size > 0:
                bin_means, _, _ = binned_statistic(
                    r_vals, intensities, statistic="mean", bins=bin_edges
                )

                # bin_means will have NaN for any bin that had 0 points.
                # We must fill these small gaps.
                nan_mask = np.isnan(bin_means)
                if np.any(nan_mask) and not np.all(nan_mask):
                    # Create an x-coordinate array for interpolation
                    x = np.arange(bin_means.size)
                    # Interpolate ONLY the nan values
                    bin_means[nan_mask] = np.interp(
                        x[nan_mask],  # points to interpolate
                        x[~nan_mask],  # known x's
                        bin_means[~nan_mask],  # known y's
                    )
                interp_vals = bin_means
            else:
                interp_vals = np.full(r_grid.shape, np.nan)
            profiles[i, :] = interp_vals

    else:
        # 3-STEP APPROACH: Fine grid → Gaussian blur → Coarse grid
        # Compute oversampling grids
        n_bins_final = int(np.ceil((max_r - min_r) * resample))
        final_r = np.linspace(min_r, max_r, n_bins_final)

        # Fine grid: oversample by 3x relative to final grid
        n_bins_fine = int(np.ceil((max_r - min_r) * resample * 100))
        fine_r = np.linspace(final_r[0], final_r[-1], n_bins_fine)

        profile_length = final_r.size
        profiles = np.full((n_angles, profile_length), np.nan, dtype=np.float32)

        for i, theta in enumerate(angles):
            r_vals, intens = _extract_wedge_radial_samples(
                phis, rs, img, theta, half_wedge, min_r, max_r
            )

            # Step 1: Fine grid resampling
            # Handle empty wedge case
            if r_vals.size > 0:
                profile_fine = np.interp(fine_r, r_vals, intens)
            else:
                profile_fine = np.full(fine_r.shape, np.nan)

            # Step 2: Gaussian smoothing on fine grid
            smooth = gaussian_filter1d(profile_fine, gaussian_sigma)

            # Step 3: Resample to coarse grid using binned statistics
            bin_edges = np.append(
                final_r,
                (
                    final_r[-1] + (final_r[1] - final_r[0])
                    if len(final_r) > 1
                    else final_r[-1] + 1
                ),
            )
            bin_means, _, _ = binned_statistic(
                fine_r, smooth, statistic="mean", bins=bin_edges
            )
            # Handle any remaining NaNs from empty bins
            nan_mask = np.isnan(bin_means)
            if np.any(nan_mask) and not np.all(nan_mask):
                x = np.arange(bin_means.size)
                bin_means[nan_mask] = np.interp(
                    x[nan_mask],
                    x[~nan_mask],
                    bin_means[~nan_mask],
                )
            profiles[i, :] = bin_means

    # Compute the derivative to obtain the sinogram
    sinogram = np.gradient(profiles, derivative_step, axis=1)

    return profiles.T, -sinogram.T


def find_best_center_shift(sinogram: np.ndarray, max_shift: int = None) -> int:
    """
    Determines the vertical shift that best centers a sinogram by symmetry minimization.

    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles) representing the sinogram.
    max_shift
        Maximum absolute shift (in rows) to test. Defaults to n_rays // 4.

    Returns
    -------
    int
        Integer shift value minimizing symmetry error.
    """
    n_rays, n_angles = sinogram.shape
    if max_shift is None:
        max_shift = n_rays // 4

    half = n_angles // 2
    shift_range = range(-max_shift, max_shift + 1)
    errors = np.zeros(2 * max_shift + 1)

    for idx, delta in enumerate(shift_range):
        # Use np.roll for pure integer shifting
        sino_shifted = np.roll(sinogram, delta, axis=0)

        if delta > 0:
            sino_valid = sino_shifted[delta:, :]
        elif delta < 0:
            sino_valid = sino_shifted[:delta, :]
        else:
            sino_valid = sino_shifted

        # Calculate symmetry error only on the valid data
        first = sino_valid[:, :half]
        second = sino_valid[:, half:]
        second_flipped = np.flip(second, axis=0)  # flip top<->bottom

        errors[idx] = np.mean((first - second_flipped) ** 2)

    # Find the delta with minimum error
    best_idx = np.argmin(errors)
    best_delta = shift_range[best_idx]
    return best_delta


def manual_center_sinogram(sinogram: np.ndarray, delta: int) -> tuple[np.ndarray, int]:
    """
    Manually centers a sinogram by applying a specified vertical shift.

    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles).
    delta
        Integer shift value to apply (positive shifts down).

    Returns
    -------
    centered : np.ndarray
        Centered sinogram array, possibly cropped symmetrically.
    delta : int
        Applied integer shift value.
    """
    # Use np.roll for pure integer shifting (no interpolation needed)
    centered = np.roll(sinogram, delta, axis=0)
    if delta > 0:
        # Shift was DOWN, fill values are at the TOP. Crop the top.
        crop = delta
        return centered[crop:, :], delta

    elif delta < 0:
        # Shift was UP, fill values are at the BOTTOM. Crop the bottom.
        crop = np.abs(delta)
        return centered[:-crop, :], delta

    else:
        # No shift, no crop
        return centered, delta


def auto_center_sinogram(
    sinogram: np.ndarray, max_shift: int = None
) -> tuple[np.ndarray, int]:
    """
    Automatically centers a sinogram by shifting it to minimize asymmetry.

    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles).
    max_shift
        Maximum absolute shift to consider. Defaults to n_rays // 4.

    Returns
    -------
    centered : np.ndarray
        Centered sinogram array, possibly cropped symmetrically.
    delta : int
        Applied integer shift value.
    """
    delta = find_best_center_shift(sinogram, max_shift=max_shift)
    return manual_center_sinogram(sinogram, delta)


def symmetrize_sinogram(sino360: np.ndarray) -> np.ndarray:
    """
    Averages a full 360° sinogram into 180° by pairing angles θ and θ+180°.

    Parameters
    ----------
    sino360
        2D array of shape (n_rays, n_angles).

    Returns
    -------
    np.ndarray
        2D array of shape (n_rays, n_angles // 2), symmetrized sinogram.
    """
    _, n_angles = sino360.shape
    assert n_angles % 2 == 0, "Need an even number of angles"
    half = n_angles // 2

    # Split into first half [0..half-1] and second half [half..]
    first = sino360[:, :half]  # This is theta = 0° to 179°
    second = sino360[:, half:]  # This is theta = 180° to 359°

    # We need to average sino(r, theta) with sino(-r, theta + 180)
    # Flipping axis 0 flips r -> -r
    second_flipped_radially = np.flip(second, axis=0)

    # Average
    sino180 = 0.5 * (first + second_flipped_radially)
    return sino180


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
        If True, average sinogram over 180° before reconstruction.

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
    # Prepare angles for full 360° sinogram
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
