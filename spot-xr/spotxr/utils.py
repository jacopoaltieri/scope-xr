import matplotlib.pyplot as plt
import numpy as np
import os
import imageio.v3 as iio


def eval_minimum_magnification(a: float, n: int, p: float) -> float:
    """Evaluate the minimum magnification required to obtain a focal spot image involving a reasonable number n of pixels."""
    m = (a + n * p) / a
    return m


def eval_minimum_radius(n: int, p: float, m: float) -> float:
    """Evaluate the minimum disk radius required to obtain a focal spot image involving a reasonable number n of pixels."""
    r = (1 + n**2) * p / (2 * m)
    return r


def crop_square_roi(
    img: np.ndarray,
    center: tuple[float, float],
    radius: float,
    width_factor: float = 1.5,
    output_path: str = None,
) -> np.ndarray:

    cx, cy = center
    half_w = int(radius * width_factor)

    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w, img.shape[1])
    y0 = max(cy - half_w, 0)
    y1 = min(cy + half_w, img.shape[0])

    cropped = img[y0:y1, x0:x1]
    if output_path is not None:
        plt.imsave(
            os.path.join(output_path, "cropped.png"),
            cropped.astype(np.uint16),
            cmap="gray",
        )
    return cropped


def save_16bit_tiff(data: np.ndarray, path: str):
    """Scales and saves a NumPy array as a 16-bit grayscale TIFF."""
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
    iio.imwrite(path, scaled_data, compression='deflate')


def interpolate_nans_1d(y):
    """
    Linearly interpolate NaNs in a 1D array.
    """
    nans = np.isnan(y)
    not_nans = ~nans
    if np.all(nans):
        # All NaN — leave as zeros or fill with a constant if you prefer
        return np.zeros_like(y)
    return np.interp(np.arange(len(y)), np.flatnonzero(not_nans), y[not_nans])


def suggest_os_angle(p: float, n: int, r: float) -> float:
    """Suggest the optimal oversampling angle (in degrees) to ensure that the cross-talk between neighboring profiles is negligible."""

    dtheta = 2*np.arccos(1-p/(n*r))
    dtheta = np.degrees(dtheta)
    return dtheta