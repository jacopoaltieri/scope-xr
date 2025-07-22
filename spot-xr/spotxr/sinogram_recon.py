import numpy as np
import tifffile
from scipy.ndimage import map_coordinates, shift, gaussian_filter1d
from skimage.transform import iradon
from spotxr.utils import interpolate_nans_1d


def _check_phl(
    img: np.ndarray, cx: float, radius: float, profile_half_length: int
) -> int:
    """
    Adjusts profile_half_length to avoid crossing image boundaries along the horizontal direction (theta = 0).

    Args:
        img: 2D grayscale image array.
        cx: X-coordinate of the circle center.
        radius: Radius of the circle.
        profile_half_length: Desired half-length of the sampling profile.

    Returns:
        adjusted_phl: Adjusted profile_half_length that fits within image bounds.
    """
    nx = 1.0
    _, img_w = img.shape

    for direction in [-1, 1]:
        d_edge = direction * profile_half_length
        px = cx + (radius + d_edge) * nx

        if not (0 <= px < img_w):
            if direction > 0:
                max_x_dist = img_w - 1 - (cx + radius * nx)
            else:
                max_x_dist = cx + radius * nx

            max_extra_length = max_x_dist / abs(nx) if abs(nx) > 1e-6 else np.inf
            new_half_length = int(min(profile_half_length, max_extra_length) - 1)

            if new_half_length < profile_half_length:
                print(
                    f"Warning: profile_half_length reduced from {profile_half_length} to {new_half_length} to avoid crossing image border."
                )
                adjusted_phl = new_half_length
            else:
                adjusted_phl = profile_half_length
    return adjusted_phl


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

    Args:
        img: 2D grayscale image array.
        cx: X-coordinate of the circle center.
        cy: Y-coordinate of the circle center.
        radius: Radius of the circle.
        n_angles: Number of angular samples around the circle.
        profile_half_length: Half-length (in pixels) of the radial sampling profile.
        derivative_step: Step size for computing the radial derivative.

    Returns:
        profiles: 2D array of shape (profile_length, n_angles), radial profiles.
        sinogram: 2D array of shape (profile_length, n_angles), negative radial derivative profiles.
    """
    profile_half_length = _check_phl(img, cx, radius, profile_half_length)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    profile_length = int(2 * profile_half_length)

    profiles = np.zeros((n_angles, profile_length), dtype=np.float32)
    sinogram = np.zeros((n_angles, profile_length), dtype=np.float32)

    for i, theta in enumerate(angles):
        # Generate unit vector pointing outward from the circle
        nx = np.cos(theta)
        ny = np.sin(theta)

        # Sample points along the normal direction
        profile = np.zeros(profile_length)
        for j in range(profile_length):
            d = j - profile_half_length
            px = cx + (radius + d) * nx
            py = cy + (radius + d) * ny

            # Interpolate pixel value
            if 0 <= int(py) < img.shape[0] and 0 <= int(px) < img.shape[1]:
                profile[j] = map_coordinates(img, [[py], [px]], order=1)[0]
            else:
                profile[j] = 0  # Zero padding if outside image

        profiles[i, :] = profile  # Store the radial profile

    # Compute the derivative to obtain the sinogram
    sinogram = np.gradient(profiles, derivative_step, axis=1)
    return profiles.T, -sinogram.T


def compute_subpixel_profiles_and_sinogram_traditional(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int,
    profile_half_length: int,
    derivative_step: int,
    dtheta: float,
    resample_radial: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes sub-pixel edge profiles and sinogram by oversampled binning in angular wedges.

    Args:
        img: 2D grayscale image array.
        cx: X-coordinate of the circle center.
        cy: Y-coordinate of the circle center.
        radius: Radius of the circle.
        n_angles: Number of angular samples (in full 360°).
        profile_half_length: Half-length (in pixels) of radial sampling.
        derivative_step: Step size for derivative computation.
        dtheta: Angular width (degrees) of wedge around each angle.
        resample_radial: Radial bin width for oversampling (in pixels).

    Returns:
        profiles: 2D array of shape (profile_bins, n_angles), radial profiles.
        sinogram: 2D array of shape (profile_bins, n_angles), negative radial derivatives.
    """
    profile_half_length = _check_phl(img, cx, radius, profile_half_length)

    # Convert angles and angular wedge width to radians
    angles = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False))
    half_wedge = np.deg2rad(dtheta) / 2

    # Coordinates relative to center
    ys, xs = np.indices(img.shape, dtype=np.float32)
    xs -= cx
    ys -= cy

    # Polar coordinates
    phis = np.arctan2(ys, xs)
    rs = np.hypot(xs, ys) - radius

    # Radial grid setup
    min_r = -profile_half_length
    max_r = profile_half_length
    n_bins = int(np.ceil((max_r - min_r) / resample_radial))
    bin_edges = np.linspace(min_r, max_r, n_bins + 1)

    # Initialize profiles array (angles x radial bins)
    profiles = np.full((n_angles, n_bins), np.nan, dtype=np.float32)

    for i, theta in enumerate(angles):
        # Angular difference wrapped to [-pi, pi]
        dphi = (phis - theta + np.pi) % (2 * np.pi) - np.pi

        # Select pixels within angular wedge
        mask = np.abs(dphi) <= half_wedge
        r_vals = rs[mask]
        intensities = img[mask]

        # Restrict to radial range
        radial_mask = (r_vals >= min_r) & (r_vals <= max_r)
        r_vals = r_vals[radial_mask]
        intensities = intensities[radial_mask]

        if r_vals.size == 0:
            continue

        # Bin radial distances
        bin_indices = np.digitize(r_vals, bin_edges) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
        bin_indices = bin_indices[valid_bins]
        intensities = intensities[valid_bins]

        if bin_indices.size == 0:
            continue

        # Compute mean intensity per bin
        counts = np.bincount(bin_indices, minlength=n_bins)
        sums = np.bincount(bin_indices, weights=intensities, minlength=n_bins)
        means = np.full(n_bins, np.nan, dtype=np.float32)
        valid_counts = counts > 0
        means[valid_counts] = sums[valid_counts] / counts[valid_counts]

        # Interpolate NaNs linearly to fill gaps
        profiles[i, :] = interpolate_nans_1d(means)

    # Compute radial derivative along radial axis (axis=1)
    sinogram = np.gradient(profiles, derivative_step, axis=1)

    return profiles.T, -sinogram.T


# https://www.researchgate.net/publication/387092230_Single-shot_2D_detector_point-spread_function_analysis_employing_a_circular_aperture_and_a_back-projection_approach
def compute_subpixel_profiles_and_sinogram_3step(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int,
    profile_half_length: int,
    derivative_step: int,
    dtheta: float,
    gaussian_sigma: float,
    resample1: float,
    resample2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes sub-pixel edge profiles and sinogram by 3step oversampled binning in angular wedges.
    Args:
        img: 2D grayscale image array.
        cx: X-coordinate of the circle center.
        cy: Y-coordinate of the circle center.
        radius: Radius of the circle.
        n_angles: Number of angular samples around the circle.
        profile_half_length: Half-length (in pixels) of radial sampling.
        derivative_step: Step size for derivative computation.
        dtheta: Angular wedge width (degrees).
        gaussian_sigma: Sigma for Gaussian smoothing on fine grid.
        resample1: Radial step for fine sampling (in pixels).
        resample2: Radial step for final subsampling (in pixels).

    Returns:
        profiles: 2D array of shape (profile_length, n_angles), oversampled radial profiles.
        sinogram: 2D array of shape (profile_length, n_angles), negative radial derivatives.
    """
    profile_half_length = _check_phl(img, cx, radius, profile_half_length)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    profile_length = 2 * profile_half_length
    delta = np.deg2rad(dtheta) / 2

    # 2) Precompute polar coords relative to circle edge
    ys, xs = np.indices(img.shape)
    xs = xs.astype(np.float32) - cx
    ys = ys.astype(np.float32) - cy
    phis = np.arctan2(ys, xs)
    rs = np.hypot(xs, ys) - radius

    # precompute oversampling grids
    final_r = np.arange(
        -profile_half_length, profile_half_length + resample2, resample2
    )

    # Fine grid used for interpolation/smoothing
    fine_r = np.arange(final_r[0], final_r[-1] + resample1, resample1)

    profile_length = final_r.size  # number of samples in radial direction
    profiles = np.zeros((n_angles, profile_length), dtype=np.float32)
    sinogram = np.zeros((n_angles, profile_length), dtype=np.float32)

    for i, theta in enumerate(angles):
        # mask pixels in angular wedge
        dphi = (phis - theta + np.pi) % (2 * np.pi) - np.pi
        mask = np.abs(dphi) <= delta
        r_vals = rs[mask]
        intens = img[mask]

        # sort and build non-uniform ESF
        idx = np.argsort(r_vals)
        r_vals = r_vals[idx]
        intens = intens[idx]

        # fine resampling
        profile_fine = np.interp(fine_r, r_vals, intens)

        # smooth with Gaussian filter
        smooth = gaussian_filter1d(profile_fine, gaussian_sigma / resample1)

        # resampling to actual subsample grid
        profile_oversampled = np.interp(final_r, fine_r, smooth)
        profiles[i, :] = profile_oversampled

    # Compute the derivative to obtain the sinogram
    sinogram = np.gradient(profiles, derivative_step, axis=1)
    return profiles.T, -sinogram.T


def find_best_center_shift(sinogram: np.ndarray, max_shift=None) -> int:
    """
    Determines the vertical shift that best centers a sinogram by symmetry minimization.

    Args:
        sinogram: 2D array of shape (n_rays, n_angles) representing the sinogram.
        max_shift: Maximum absolute shift (in rows) to test. Defaults to n_rays // 4.

    Returns:
        best_delta: Integer shift value minimizing symmetry error.
    """
    n_rays, n_angles = sinogram.shape
    if max_shift is None:
        max_shift = n_rays // 4

    half = n_angles // 2
    errors = {}
    for delta in range(-max_shift, max_shift + 1):
        # shift the sinogram up/down
        sino_shifted = shift(sinogram, shift=[delta, 0], order=1, mode="nearest")

        first = sino_shifted[:, :half]
        second = sino_shifted[:, half:]
        second_flipped = np.flip(second, axis=0)  # flip top<->bottom

        # compute mean squared difference
        err = np.mean((first - second_flipped[:, ::-1]) ** 2)
        errors[delta] = err

    # pick the delta with minimum error
    best_delta = min(errors, key=errors.get)
    return best_delta


def auto_center_sinogram(
    sinogram: np.ndarray, max_shift=None
) -> tuple[np.ndarray, int]:
    """
    Automatically centers a sinogram by shifting it to minimize asymmetry.

    Args:
        sinogram: 2D array of shape (n_rays, n_angles).
        max_shift: Maximum absolute shift to consider. Defaults to n_rays // 4.

    Returns:
        centered: Centered sinogram array, possibly cropped symmetrically.
        delta: Applied integer shift value.
    """
    delta = find_best_center_shift(sinogram, max_shift=max_shift)
    centered = shift(sinogram, shift=[delta, 0], order=3, mode="nearest")
    if delta != 0:
        # Crop symmetric margins
        crop = np.abs(delta)
        return centered[crop:-crop, :], delta
    else:
        return centered, delta


def symmetrize_sinogram(sino360: np.ndarray) -> np.ndarray:
    """
    Averages a full 360° sinogram into 180° by pairing angles θ and θ+180°.

    Args:
        sino360: 2D array of shape (n_rays, 360).

    Returns:
        sino180: 2D array of shape (n_rays, 180), symmetrized sinogram.
    """
    n_rays, n_angles = sino360.shape
    assert n_angles % 2 == 0, "Need an even number of angles"
    half = n_angles // 2

    # Split into first half [0..half-1] and second half [half..]
    first = sino360[:, :half]
    second = sino360[:, half:]
    # Flip second in the angular axis so that θ+180 lines up with θ
    second_flipped = np.flip(second, axis=1)
    # Average
    sino180 = 0.5 * (first + second_flipped)
    return sino180


def reconstruct_focal_spot(
    sinogram: np.ndarray, filter_name: str, symmetrize: bool
) -> np.ndarray:
    """
    Reconstructs the focal spot image from sinogram via filtered back-projection.

    Args:
        sinogram: 2D array of shape (n_rays, n_angles).
        filter_name: Name of the filter to use in the inverse radon transform.
        symmetrize: If True, average sinogram over 180° before reconstruction.

    Returns:
        reconstruction: 2D array representing the reconstructed focal spot.
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

    Args:
        sinogram: 2D array of shape (n_rays, n_angles).
        output_tiff_path: Path for the output multi-page TIFF file.
        filter_name: Filter name for the inverse radon transform.
        shifts: List of integer shifts (rows) to apply to sinogram.
    """
    reconstructions = []
    # Prepare angles for full 360° sinogram
    n_angles = sinogram.shape[1]
    theta = np.linspace(0.0, 360.0, n_angles, endpoint=False)

    for delta in shifts:
        # shift sinogram vertically:
        #   shifting by +delta moves content down, so the effective axis moves up
        shifted_sino = shift(sinogram, shift=[delta, 0], order=1, mode="nearest")

        # reconstruct
        rec = iradon(shifted_sino, theta=theta, filter_name=filter_name)
        reconstructions.append(rec.astype(np.float32))

    tifffile.imwrite(
        output_tiff_path, np.stack(reconstructions, axis=0), photometric="minisblack"
    )
