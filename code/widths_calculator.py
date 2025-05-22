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
        slopes: 1D array of computed slopes for each profile.
    """
    n_rays, n_angles = profiles.shape
    x = np.arange(n_rays)
    slopes = np.zeros(n_angles, dtype=float)

    p0 = [profiles.max() - profiles.min(), n_rays / 2, n_rays / 8, profiles.min()]

    for i in range(n_angles):
        p = profiles[:, i]
        try:
            popt, _ = curve_fit(erf_step, x, p, p0=p0, maxfev=2000)
            A, x0, sigma, B = popt
            slope = A / (np.sqrt(np.pi) * sigma)
        except RuntimeError:
            slope = 0
        slopes[i] = slope

    wide_idx = int(np.argmax(slopes))
    narrow_idx = int(np.argmin(slopes))
    return wide_idx, narrow_idx, slopes
