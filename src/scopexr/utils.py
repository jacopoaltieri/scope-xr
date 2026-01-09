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

from pathlib import Path
from typing import Optional, Callable
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


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
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Crop a square region of interest (ROI) around the specified center.

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
        The cropped image array.
    """
    cx, cy = center
    half_w = int(radius * width_factor)

    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w, img.shape[1])
    y0 = max(cy - half_w, 0)
    y1 = min(cy + half_w, img.shape[0])

    cropped = img[int(y0) : int(y1), int(x0) : int(x1)]

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
    # 1. Normalize the data to the 0-1 range
    data_min = data.min()
    data_max = data.max()

    if data_max == data_min:
        # Handle constant images (scale to 0 or 65535, depending on value)
        if data_min == 0:
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = np.ones_like(data)
    else:
        normalized_data = (data - data_min) / (data_max - data_min)

    # 2. Scale to 0-65535 and convert to uint16
    # Rounding is important before conversion
    scaled_data = np.round(normalized_data * 65535).astype(np.uint16)

    # 3. Save using imageio with lossless compression
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
        # All NaN — leave as zeros or fill with a constant if you prefer
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
    out_dir: str,
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
