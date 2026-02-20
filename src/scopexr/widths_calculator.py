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

import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def fwhm(profile: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the Full Width at Half Maximum (FWHM) of a 1D profile using linear interpolation.

    Parameters
    ----------
    profile
        1D array representing a single profile (e.g. from a sinogram).

    Returns
    -------
    width : float
        Width in pixels between half-maximum crossings (interpolated).
    left_idx : float
        Fractional index of left half-maximum crossing.
    right_idx : float
        Fractional index of right half-maximum crossing.
    """
    profile = np.asarray(profile)
    n = len(profile)

    peak_idx = np.argmax(profile)
    peak_val = profile[peak_idx]
    half_max = 0.5 * peak_val

    # --- Find left half-maximum crossing ---
    left_idx = None
    for i in range(peak_idx, 0, -1):
        if profile[i] >= half_max > profile[i - 1]:
            denominator = profile[i] - profile[i - 1]
            if denominator == 0:
                frac = 0.5  # avoid division by zero
            else:
                frac = (half_max - profile[i - 1]) / denominator

            left_idx = (i - 1) + frac
            break

    # --- Find right half-maximum crossing ---
    right_idx = None
    for i in range(peak_idx, n - 1):
        if profile[i] >= half_max > profile[i + 1]:
            frac = (half_max - profile[i]) / (profile[i + 1] - profile[i])
            right_idx = i + frac
            break

    if left_idx is None or right_idx is None:
        return np.nan, np.nan, np.nan  # no valid FWHM found

    width = right_idx - left_idx
    return width, left_idx, right_idx


def fw15m(profile: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the Full Width at 15 Percent Maximum (FW15M / 15% max) of a 1D profile using linear interpolation.

    Parameters
    ----------
    profile
        1D array representing a single profile (e.g. from a sinogram).

    Returns
    -------
    width : float
        Width in pixels between 15% maximum crossings (interpolated).
    left_idx : float
        Fractional index of left 15% maximum crossing.
    right_idx : float
        Fractional index of right 15% maximum crossing.
    """
    profile = np.asarray(profile)
    n = len(profile)

    # Find peak and 15% maximum (baseline subtraction implies min ≈ 0)
    peak_idx = np.argmax(profile)
    peak_val = profile[peak_idx]
    fifteen_max = 0.15 * peak_val

    # --- Find left crossing ---
    left_idx = None
    for i in range(peak_idx, 0, -1):
        if profile[i] >= fifteen_max > profile[i - 1]:
            denominator = profile[i] - profile[i - 1]
            if denominator == 0:
                frac = 0.5  # avoid division by zero
            else:
                frac = (fifteen_max - profile[i - 1]) / denominator

            left_idx = (i - 1) + frac
            break

    # --- Find right crossing ---
    right_idx = None
    for i in range(peak_idx, n - 1):
        if profile[i] >= fifteen_max > profile[i + 1]:
            frac = (fifteen_max - profile[i]) / (profile[i + 1] - profile[i])
            right_idx = i + frac
            break

    if left_idx is None or right_idx is None:
        return np.nan, np.nan, np.nan  # no valid FWTM found

    width = right_idx - left_idx
    return width, left_idx, right_idx


def fwhm_from_sigma(sigma: float) -> float:
    """
    Compute the FWHM from the standard deviation of an error function step.

    Parameters
    ----------
    sigma
        Standard deviation of the error function step.

    Returns
    -------
    float
        FWHM value.
    """
    return 2 * sigma * np.sqrt(2 * np.log(2))


def erf_step(x: np.ndarray, a: float, x0: float, sigma: float, b: float) -> np.ndarray:
    """
    Error function step model for fitting profile edges.

    Parameters
    ----------
    x
        Independent variable (e.g., pixel positions).
    A
        Amplitude of the step.
    x0
        Center position of the transition.
    sigma
        Width of the transition (standard deviation).
    B
        Background offset.

    Returns
    -------
    np.ndarray
        Model values evaluated at x.
    """
    return a * erf((x - x0) / sigma) + b


def find_extreme_profiles_erf(profiles: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    Fit each angular profile to an error function step and find extreme slopes.

    Parameters
    ----------
    profiles
        2D array of shape [n_rays, n_angles] containing line profiles.

    Returns
    -------
    wide_idx : int
        Index of the profile with the steepest (widest) slope.
    narrow_idx : int
        Index of the profile with the shallowest (narrowest) slope.
    sigmas : np.ndarray
        1D array of computed sigmas for each profile.
    """
    n_rays, n_angles = profiles.shape
    x = np.arange(n_rays)
    sigmas = np.zeros(n_angles, dtype=float)

    for i in range(n_angles):
        profile = average_neighbors(profiles, i, 3)

        profile_min = profile.min()
        profile_max = profile.max()
        peak_idx = np.argmax(profile)

        p0 = [
            profile_max - profile_min,  # amplitude
            peak_idx,  # center position
            n_rays / 8,  # sigma
            profile_min,  # baseline
        ]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(erf_step, x, profile, p0=p0, maxfev=1000)
            sigmas[i] = popt[2]

        except RuntimeError:
            # Fit failed: record nan and a placeholder
            sigmas[i] = np.nan

    wide_idx = int(np.argmax(sigmas))
    narrow_idx = int(np.argmin(sigmas))

    # If fitting failed to find meaningful extreme profiles, use indices 0 and 89
    if np.all(sigmas == 0) or np.all(np.isnan(sigmas)):
        print(
            "Warning: Curve fitting failed to find extreme profiles. Using angles 0 and 90 as fallback."
        )
        wide_idx = 89
        narrow_idx = 0

    return wide_idx, narrow_idx, sigmas


def average_neighbors(
    sinogram: np.ndarray, angle_idx: int, line_width: int
) -> np.ndarray:
    """
    Compute the vertical profile at a given angle index, averaging across multiple adjacent rows.

    Parameters
    ----------
    sinogram
        2D sinogram array of shape (rows/pixels, angles).
    angle_idx
        The angle (column index) to extract the profile from.
    line_width
        Number of adjacent rows to average (must be odd).

    Returns
    -------
    np.ndarray
        A 1D profile averaged across multiple rows.
    """
    assert line_width % 2 == 1, "line_width must be odd"

    if line_width == 1:
        # Fast path for no averaging
        return sinogram[:, angle_idx]

    half_width = line_width // 2
    rows, _ = sinogram.shape

    # Pre-allocate output array
    profile_stack = np.zeros((line_width, rows), dtype=sinogram.dtype)

    # Vectorized approach: compute all row indices at once
    base_indices = np.arange(rows)
    for i, offset in enumerate(range(-half_width, half_width + 1)):
        row_idx = np.clip(base_indices + offset, 0, rows - 1)
        profile_stack[i] = sinogram[row_idx, angle_idx]

    return np.mean(profile_stack, axis=0)


def compute_fs_width(
    fwhm_px: float, pixel_size: float, fs_magnification: float
) -> float:
    """
    Compute the focal spot width in micrometers from the FWHM in pixels.

    Parameters
    ----------
    fwhm_px
        Full width at half maximum in pixels.
    pixel_size
        Size of a pixel in micrometers.
    fs_magnification
        Magnification factor of the focal spot.

    Returns
    -------
    float
        Focal spot width in micrometers.
    """
    return fwhm_px * pixel_size / fs_magnification


def gaussian(x: np.ndarray, a: float, mu: float, sigma: float, b: float) -> np.ndarray:
    """
    Gaussian model for fitting sinusoidal profiles.

    Parameters
    ----------
    x
        Independent variable (e.g., pixel positions).
    A
        Amplitude of the Gaussian.
    mu
        Mean (center) of the Gaussian.
    sigma
        Standard deviation of the Gaussian.
    B
        Baseline offset.

    Returns
    -------
    np.ndarray
        Gaussian curve evaluated at x.
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + b


def find_extreme_profiles_gaussian(
    sinogram: np.ndarray,
) -> tuple[int, int, np.ndarray, list[np.ndarray]]:
    """
    Fit each sinogram profile (column) to a Gaussian curve and extract the extreme profiles.

    Parameters
    ----------
    sinogram
        2D array of shape (n_rays, n_angles) containing line profiles.

    Returns
    -------
    wide_idx : int
        Index of the profile with the largest sigma (widest).
    narrow_idx : int
        Index of the profile with the smallest sigma (narrowest).
    sigmas : np.ndarray
        1D array of sigma values for each profile.
    popts : list[np.ndarray]
        List of optimal fit parameters [A, mu, sigma, B] for each profile;
        entries are np.array([nan, nan, nan, nan]) on fit failure.
    """
    n_rays, n_angles = sinogram.shape
    x = np.arange(n_rays)
    sigmas = np.zeros(n_angles, dtype=float)
    popts: list[np.ndarray] = []

    # Pre-compute bounds once (they're constant)
    bounds = (
        [0, 0, 1e-6, -np.inf],  # Lower bounds: A, mu, sigma, B
        [np.inf, n_rays, n_rays, np.inf],  # Upper bounds
    )

    for i in range(n_angles):
        profile = average_neighbors(sinogram, i, 3)

        profile_max = profile.max()
        profile_min = profile.min()
        peak_idx = np.argmax(profile)

        p0 = [
            profile_max - profile_min,  # amplitude
            peak_idx,  # mean
            n_rays / 8,  # sigma
            profile_min,  # baseline
        ]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    gaussian,
                    x,
                    profile,
                    p0=p0,
                    bounds=bounds,
                    maxfev=1000,  # Reduced from 2000
                )
            popts.append(popt)
            sigmas[i] = popt[2]
        except RuntimeError:
            # Fit failed: record nan and a placeholder
            sigmas[i] = np.nan
            popts.append(np.array([np.nan, np.nan, np.nan, np.nan]))

    wide_idx = int(np.nanargmax(sigmas))
    narrow_idx = int(np.nanargmin(sigmas))

    # If fitting failed to find meaningful extreme profiles, use indices 0 and 89
    if np.all(np.isnan(sigmas)):
        print(
            "Warning: Curve fitting failed to find extreme profiles. Using angles 0 and 90 as fallback."
        )
        wide_idx = 89
        narrow_idx = 0

    return wide_idx, narrow_idx, sigmas, popts
