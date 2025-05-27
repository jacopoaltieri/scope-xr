import numpy as np
import tifffile
from skimage.transform import iradon
from scipy.ndimage import map_coordinates, shift


def compute_profiles_and_sinogram(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int = 360,
    profile_half_length: int = 64,
    derivative_step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract edge profiles around a circle and compute the sinogram.

    Args:
        cropped: 2D ndarray (grayscale image)
        cx, cy: center of the circle
        radius: radius of the circle
        n_angles: number of angles to sample (default 360)
        profile_half_length: number of pixels to sample on either side of the edge (default 64)
        derivative_step: step size for computing the derivative (default 1)
    Returns:
        profiles: 2D array of extracted profiles for visualization
        sinogram: 2D array [angle_index, radial_profile]
    """
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    profile_length = 2 * profile_half_length

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


def find_best_center_shift(sinogram: np.ndarray, max_shift=None) -> int:
    """
    Find the vertical shift (in rows) that best centers the sinogram,
    by minimizing the symmetry error between the first 180° and flipped 180°.

    Parameters
    ----------
    sinogram: 2D ndarray, shape (n_rays, n_angles)
    max_shift: Maximum absolute shift (in rows) to try. Defaults to n_rays//4.

    Returns
    -------
    best_delta: The integer vertical shift that yields lowest symmetry error.
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
    Auto center the sinogram by shifting it by the best symmetry based offset.

    Returns the centered sinogram and the applied shift.
    """
    delta = find_best_center_shift(sinogram, max_shift=max_shift)
    centered = shift(sinogram, shift=[delta, 0], order=3, mode="nearest")
    return centered, delta


def symmetrize_sinogram(sino360: np.ndarray) -> np.ndarray:
    """
    Takes sino360 of shape (n_rays, 360) and returns
    sino180 of shape (n_rays, 180) after averaging θ with θ+180.
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
    sinogram: np.ndarray, filter_name:str, symmetrize:bool
) -> np.ndarray:
    """
    Reconstruct the focal spot from the sinogram using filtered back-projection.

    Args:
        sinogram: 2D array [angle_index, radial_profile]
        symmetrize: If True, symmetrize the sinogram before reconstruction.
    Returns:
        reconstruction: 2D array representing the focal spot
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
    filter_name:str,
    shifts: list,
) -> None:
    """
    For each vertical shift in `shifts`, shift the sinogram, reconstruct,
    and save all reconstructions in a single multi-page TIFF.

    Parameters
    ----------
    sinogram: 2D array [angle_index, radial_profile]
    output_tiff_path: Path to the output multi-page TIFF.
    shifts: Amounts (in rows) to shift the sinogram: positive shifts move the axis downward.
    filter_name: Filter to use in iradon.
    circle: Pass-through to skimage.transform.iradon.
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
    print(f"Saved {len(shifts)} reconstructions to '{output_tiff_path}'")
