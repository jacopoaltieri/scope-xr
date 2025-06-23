import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit


def fwhm(sinogram: np.ndarray) -> tuple[int, int, int]:
    """
    Compute the full width at half maximum (FWHM) of a 1D sinogram profile.

    Args:
        sinogram: 1D array representing a single profile from the sinogram.

    Returns:
        width_px: Width in pixels between half-maximum crossings.
        left_idx: Index of the left half-maximum crossing.
        right_idx: Index of the right half-maximum crossing.
    """
    half_max = (sinogram.max() + sinogram.min()) / 2.0
    n = len(sinogram)
    center = n // 2

    left_edge = None
    for i in range(center, -1, -1):
        if sinogram[i] < half_max:
            left_edge = i
            break
    left = (left_edge + 1) if left_edge is not None else 0

    right_edge = None
    for i in range(center, n):
        if sinogram[i] < half_max:
            right_edge = i
            break
    right = (right_edge - 1) if right_edge is not None else (n - 1)

    width = right - left
    return width, left, right


def fwhm_from_sigma(sigma: float) -> float:
    """
    Compute the FWHM from the standard deviation of an error function step.

    Args:
        sigma: Standard deviation of the error function step.

    Returns:
        FWHM value.
    """
    return 2 * sigma * np.sqrt(2 * np.log(2))


def erf_step(x: np.ndarray, A: float, x0: float, sigma: float, B: float) -> np.ndarray:
    """
    Error function step model for fitting profile edges.

    Args:
        x: Independent variable (e.g., pixel positions).
        A: Amplitude of the step.
        x0: Center position of the transition.
        sigma: Width of the transition (standard deviation).
        B: Background offset.

    Returns:
        Model values evaluated at x.
    """
    return A * erf((x - x0) / sigma) + B


def find_extreme_profiles_erf(profiles: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    Fit each angular profile to an error function step and compute its slope.

    Args:
        profiles: 2D array of shape [n_rays, n_angles] containing line profiles.

    Returns:
        wide_idx: Index of the profile with the steepest (widest) slope.
        narrow_idx: Index of the profile with the shallowest (narrowest) slope.
        sigmas: 1D array of computed sigmas for each profile.
    """
    n_rays, n_angles = profiles.shape
    x = np.arange(n_rays)
    sigmas = np.zeros(n_angles, dtype=float)

    p0 = [profiles.max() - profiles.min(), n_rays / 2, n_rays / 8, profiles.min()]

    for i in range(n_angles):
        profile = average_neighbors(profiles, i)
        try:
            popt, _ = curve_fit(erf_step, x, profile, p0=p0, maxfev=2000)
            A, x0, sigma, B = popt
            slope = A / (np.sqrt(np.pi) * sigma)
        except RuntimeError:
            slope = 0
        sigmas[i] = sigma

    wide_idx = int(np.argmax(sigmas))
    narrow_idx = int(np.argmin(sigmas))
    return wide_idx, narrow_idx, sigmas


def average_neighbors(
    sinogram: np.ndarray, angle_idx: int, line_width: int = 3
) -> np.ndarray:
    """
    Compute the vertical profile at a given angle index, averaging across multiple adjacent rows.

    Args:
        sinogram: 2D sinogram array of shape (rows/pixels, angles).
        angle_idx: The angle (column index) to extract the profile from.
        line_width: Number of adjacent rows to average (must be odd).

    Returns:
        A 1D profile averaged across multiple rows.
    """
    assert line_width % 2 == 1, "line_width must be odd"
    half_width = line_width // 2
    rows, _ = sinogram.shape

    # Stack rows centered at each position and average
    profile_stack = []
    for offset in range(-half_width, half_width + 1):
        row_idx = np.clip(np.arange(rows) + offset, 0, rows - 1)  # clamp to bounds
        profile_stack.append(sinogram[row_idx, angle_idx])

    return np.mean(profile_stack, axis=0)


def compute_fs_width(fwhm_px: int, pixel_size: float, fs_magnification: float) -> float:
    """
    Compute the focal spot width in micrometers from the FWHM in pixels.

    Args:
        fwhm_px: Full width at half maximum in pixels.
        pixel_size: Size of a pixel in micrometers.
        fs_magnification: Magnification factor of the focal spot

    Returns:
        Focal spot width in micrometers.
    """
    return fwhm_px * pixel_size / fs_magnification


def gaussian(x: np.ndarray, A: float, mu: float, sigma: float, B: float) -> np.ndarray:
    """
    Gaussian model for fitting sinusoidal profiles.

    Args:
        x: Independent variable (e.g., pixel positions).
        A: Amplitude of the Gaussian.
        mu: Mean (center) of the Gaussian.
        sigma: Standard deviation of the Gaussian.
        B: Baseline offset.

    Returns:
        Gaussian curve evaluated at x.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + B


def find_extreme_profiles_gaussian(
    sinogram: np.ndarray,
) -> tuple[int, int, np.ndarray, list[np.ndarray]]:
    """
    Fit each sinogram profile (column) to a Gaussian curve and extract the extreme profiles.

    Args:
        sinogram: 2D array of shape (n_rays, n_angles) containing line profiles.

    Returns:
        wide_idx: Index of the profile with the largest sigma (widest).
        narrow_idx: Index of the profile with the smallest sigma (narrowest).
        sigmas: 1D array of computed sigma values for each profile.
        popts:  List of optimal fit parameters [A, mu, sigma, B] for each profile;
                entries are np.array([nan, nan, nan, nan]) on fit failure.
    """
    n_rays, n_angles = sinogram.shape
    x = np.arange(n_rays)
    sigmas = np.zeros(n_angles, dtype=float)
    popts: list[np.ndarray] = []

    # Initial guess: [amplitude, mean, sigma, baseline]
    p0 = [sinogram.max() - sinogram.min(), n_rays / 2, n_rays / 8, sinogram.min()]

    for i in range(n_angles):
        profile = average_neighbors(sinogram, i)
        try:
            popt, _ = curve_fit(
                gaussian,
                x,
                profile,
                p0=p0,
                bounds=(
                    [0, 0, 1e-6, -np.inf],  # Lower bounds: A, mu, sigma, B
                    [np.inf, n_rays, n_rays, np.inf],  # Upper bounds
                ),
                maxfev=2000,
            )
            popts.append(popt)
            sigmas[i] = popt[2]
        except RuntimeError:
            # Fit failed: record nan and a placeholder
            sigmas[i] = np.nan
            popts.append(np.array([np.nan, np.nan, np.nan, np.nan]))

    wide_idx = int(np.nanargmax(sigmas))
    narrow_idx = int(np.nanargmin(sigmas))
    return wide_idx, narrow_idx, sigmas, popts
