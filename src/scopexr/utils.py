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

from pathlib import Path
from typing import Optional, Callable
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def eval_minimum_magnification(a: float, n: int, p: float) -> float:
    """
    Evaluate the minimum magnification required to obtain a focal spot image involving a reasonable number n of pixels.

    Parameters
    ----------
    a
        Focal spot size (dimension).
    n
        Number of pixels desired.
    p
        Pixel size/pitch.

    Returns
    -------
    float
        The calculated minimum magnification.
    """
    m = (a + n * p) / a
    return m


def eval_minimum_radius(n: int, p: float, m: float) -> float:
    """
    Evaluate the minimum disk radius required to obtain a focal spot image involving a reasonable number n of pixels.

    Parameters
    ----------
    n
        Number of pixels desired.
    p
        Pixel size/pitch.
    m
        Magnification factor.

    Returns
    -------
    float
        The calculated minimum radius.
    """
    r = (1 + n**2) * p / (2 * m)
    return r


def crop_square_roi(
    img: np.ndarray,
    center: tuple[float, float],
    radius: float,
    width_factor: float = 1.5,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    Crop a square region of interest (ROI) around the specified center.
    If the ideal crop window extends outside the image frame, falls back to
    returning the whole image.

    Parameters
    ----------
    img
        Input image array.
    center
        (x, y) coordinates of the center.
    radius
        Radius of the feature to crop around.
    width_factor
        Factor to determine the crop size relative to the radius.
    output_path
        If provided, saves the cropped image to this directory.

    Returns
    -------
    np.ndarray
        The cropped image array (fallback to the full image if crop extends outside bounds).
    """
    cx, cy = center
    half_w = int(radius * width_factor)

    # 1. Calculate ideal box edges before checking boundaries
    ideal_x0 = int(cx) - half_w
    ideal_x1 = int(cx) + half_w
    ideal_y0 = int(cy) - half_w
    ideal_y1 = int(cy) + half_w
    print(
        f"Cropping ROI: ideal box edges = ({ideal_x0}, {ideal_y0}) to ({ideal_x1}, {ideal_y1})"
    )
    # 2. Check if the ideal crop box spills outside the image parameters
    if (
        ideal_x0 < 0
        or ideal_x1 > img.shape[1]
        or ideal_y0 < 0
        or ideal_y1 > img.shape[0]
    ):

        # Fallback to using the entire image frame
        # Note: using .copy() protects the original array if downstream functions modify it
        cropped = img.copy()

    else:
        # 3. Perform the standard slice if it fits cleanly inside the image bounds
        cropped = img[int(ideal_y0) : int(ideal_y1), int(ideal_x0) : int(ideal_x1)]

    # 4. Save the resulting image array safely
    if output_path is not None:
        plt.imsave(
            Path(output_path) / "cropped.png",
            cropped.astype(np.uint16),
            cmap="gray",
        )

    return cropped


def save_16bit_tiff(data: np.ndarray, path: str) -> None:
    """
    Scales and saves a NumPy array as a 16-bit grayscale TIFF.

    Parameters
    ----------
    data
        Input image data.
    path
        Output file path.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min

    if data_range == 0:
        # Handle constant images - use zeros for zero value, otherwise use max
        scaled_data = (
            np.zeros(data.shape, dtype=np.uint16)
            if data_min == 0
            else np.full(data.shape, 65535, dtype=np.uint16)
        )
    else:
        # Normalize and scale in one operation to avoid intermediate array
        # Use clip to handle any floating point edge cases
        scaled_data = np.clip(
            np.round((data - data_min) * (65535.0 / data_range)), 0, 65535
        ).astype(np.uint16)

    # Save using imageio with lossless compression
    iio.imwrite(path, scaled_data, compression="deflate")


def interpolate_nans_1d(y: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaNs in a 1D array.

    Parameters
    ----------
    y
        1D input array possibly containing NaNs.

    Returns
    -------
    np.ndarray
        Array with NaNs filled by linear interpolation.
    """
    nans = np.isnan(y)
    not_nans = ~nans
    if np.all(nans):
        return np.zeros_like(y)
    return np.interp(np.arange(len(y)), np.flatnonzero(not_nans), y[not_nans])


def suggest_os_angle(p: float, n: int, r: float) -> float:
    """
    Suggest the optimal oversampling angle to ensure negligible cross-talk.

    Parameters
    ----------
    p
        Pixel size.
    n
        Oversampling factor (or similar parameter depending on strategy).
    r
        Radius.

    Returns
    -------
    float
        Suggested oversampling angle in degrees.
    """
    dtheta = 2 * np.arccos(1 - p / (n * r))
    dtheta = np.degrees(dtheta)
    return dtheta


def save_and_plot(
    name: str,
    arr: np.ndarray,
    out_dir: str | Path,
    plot_func: Optional[Callable] = None,
    suffix: str = "",
    show_plots: bool = False,
) -> str:
    """
    Save a 2D array as a 16-bit TIFF and optionally plot it using a provided plotting function.

    Parameters
    ----------
    name
        Base name for the file.
    arr
        Image array to save.
    out_dir
        Output directory.
    plot_func
        Optional function to generate a plot.
    suffix
        Suffix to append to the filename.
    show_plots
        If True, show the plot interactively.

    Returns
    -------
    str
        Path to the saved TIFF file.
    """
    fname = f"{name}{suffix}.tiff" if not name.endswith(".tiff") else name
    path = Path(out_dir) / fname
    save_16bit_tiff(arr, str(path))

    if plot_func:
        plot_func(arr, out_dir, show_plots)

    return str(path)


def background_percentile(profile: np.ndarray, low_frac: float = 0.15) -> float:
    """
    Estimate background intensity from the lower percentile of a profile.

    Parameters
    ----------
    profile
        1D array representing a profile.
    low_frac
        Fraction (0-1) to use for percentile calculation. Default is 0.5 (50th percentile).

    Returns
    -------
    float
        Mean intensity of values below the percentile threshold.
    """
    profile = np.asarray(profile)
    threshold = np.percentile(profile, low_frac * 100)
    return float(np.mean(profile[profile <= threshold]))


def fit_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    sigma_initial: float,
) -> tuple[float, float, float, float]:
    def gaussian_fit(
        x: np.ndarray,
        A: float,
        mu: float,
        sigma: float,
        B: float,
    ) -> np.ndarray:
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + B

    default_params = (
        float(np.max(y)),
        0.0,
        sigma_initial,
        0.0,
    )

    try:
        popt, _ = curve_fit(
            gaussian_fit,
            x,
            y,
            p0=default_params,
            maxfev=5000,
        )

        return (
            float(popt[0]),
            float(popt[1]),
            float(popt[2]),
            float(popt[3]),
        )

    except Exception:
        return default_params
