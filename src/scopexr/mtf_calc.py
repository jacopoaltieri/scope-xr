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


def _extrapolate_lsf_tails(
    lsf: np.ndarray, threshold_ratio: float = 0.1, fit_points: int = 5
) -> np.ndarray:
    """
    Extrapolate the tails of a Line Spread Function (LSF) using an exponential fit.

    This method follows the approach described by K. Doi, where the tails of the LSF
    are extended with an exponential decay to reduce truncation artifacts. This is
    particularly useful when computing the MTF via Fourier transform, as abrupt
    truncation of the LSF can introduce low-frequency oscillations.

    Parameters
    ----------
    lsf : np.ndarray
        1D array representing the line spread function. It is assumed to have a
        single dominant peak.
    threshold_ratio : float, optional
        Fraction of the peak LSF value used to define the boundary between the
        central region and the tails. Extrapolation starts where the LSF falls
        below `threshold_ratio * max(lsf)`. Default is 0.01.
    fit_points : int, optional
        Number of samples (per side) used to fit the exponential decay in the
        log-domain near the threshold. Must be large enough for a stable fit.
        Default is 5.

    Returns
    -------
    np.ndarray
        A copy of the input LSF with left and right tails replaced by exponential
        extrapolations where applicable.
    """

    peak_index = np.argmax(lsf)
    peak_value = lsf[peak_index]
    threshold = threshold_ratio * peak_value

    lsf_extrapolated = lsf.copy()
    lsf_safe = np.clip(lsf, 1e-10, None)  # avoids log(0) and negatives

    # --- Left tail ---
    left_boundary = peak_index
    while left_boundary > 0 and lsf[left_boundary] > threshold:
        left_boundary -= 1

    left_join = left_boundary + 1
    if left_boundary > 0 and (left_join + fit_points) <= len(lsf):
        x_fit = np.arange(left_join, left_join + fit_points)
        y_fit = lsf_safe[left_join : left_join + fit_points]

        slope, _ = np.polyfit(x_fit - left_join, np.log(y_fit), 1)

        if slope > 0:  # valid decay direction for left side
            x_tail = np.arange(0, left_join)
            lsf_extrapolated[:left_join] = lsf_safe[left_join] * np.exp(
                slope * (x_tail - left_join)
            )
        else:
            lsf_extrapolated[:left_join] = 0.0

    # --- Right tail ---
    right_boundary = peak_index
    n = len(lsf)

    while right_boundary < n - 1 and lsf[right_boundary] > threshold:
        right_boundary += 1

    right_join = right_boundary - 1
    if right_boundary < n - 1 and (right_join - fit_points + 1) >= 0:
        x_fit = np.arange(right_join - fit_points + 1, right_join + 1)
        y_fit = lsf_safe[right_join - fit_points + 1 : right_join + 1]

        slope, _ = np.polyfit(x_fit - x_fit[0], np.log(y_fit), 1)

        if slope < 0:  # valid decay direction for right side
            x_tail = np.arange(right_join + 1, n)
            lsf_extrapolated[right_join + 1 :] = lsf_safe[right_join] * np.exp(
                slope * (x_tail - right_join)
            )
        else:
            lsf_extrapolated[right_join + 1 :] = 0.0

    return lsf_extrapolated


def compute_1d_mtf(
    lsf: np.ndarray, pixel_size: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute 1D MTF from LSF.

    Parameters
    ----------
    lsf
        1D array representing the line spread function (LSF).
    pixel_size
        Pixel size in mm.

    Returns
    -------
    freq: np.ndarray
        Frequencies in cycles/mm.
    mtf_1d: np.ndarray
        1D MTF array normalized to 1 at zero frequency.
    mtf10: float
        Frequency at which MTF drops to 10% (cycles/mm).
    """
    # lsf = lsf / np.sum(lsf) if np.sum(lsf) != 0 else lsf

    lsf = _extrapolate_lsf_tails(lsf, threshold_ratio=0.05, fit_points=5)
    # lsf = np.hanning(lsf.size) * lsf  # Apply Hanning window to reduce edge effects

    # Compute FFT and frequencies
    otf_1d = np.fft.fft(np.fft.ifftshift(lsf))
    mtf_1d = np.abs(otf_1d)
    mtf_1d = mtf_1d / mtf_1d[0]  # Normalize to 1 at zero frequency

    freq = np.fft.fftfreq(lsf.size, d=pixel_size)

    # Shift for plotting
    mtf_1d = np.fft.fftshift(mtf_1d)
    freq = np.fft.fftshift(freq)

    # Consider only positive frequencies
    mask = freq >= 0
    freq_pos = freq[mask]
    mtf_pos = mtf_1d[mask]

    # Find MTF10 (interpolating if needed)
    mtf10_value = 0.10
    if np.any(mtf_pos <= mtf10_value):
        idx = np.where(mtf_pos <= mtf10_value)[0][0]
        # Linear interpolation
        if idx == 0:
            mtf10_freq = freq_pos[0]
        else:
            f1, f2 = freq_pos[idx - 1], freq_pos[idx]
            m1, m2 = mtf_pos[idx - 1], mtf_pos[idx]
            mtf10_freq = f1 + (mtf10_value - m1) * (f2 - f1) / (m2 - m1)
    else:
        mtf10_freq = np.nan  # Not reached

    return freq_pos, mtf_pos, mtf10_freq


def compute_1d_mtf_from_sino(
    sinogram: np.ndarray, pixel_size: float, angle: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute 1D MTF from sinogram

    Parameters
    ----------
    sinogram
        2D array representing the sinogram.
    pixel_size
        Pixel size in mm.
    angle
        Angle index along which to extract the profile.

    Returns
    -------
    freq: np.ndarray
        Frequencies in cycles/mm.
    mtf_1d: np.ndarray
        1D MTF array normalized to 1 at zero frequency.
    mtf10: float
        Frequency at which MTF drops to 10% (cycles/mm).
    """
    lsf = sinogram[:, angle]

    lsf = _extrapolate_lsf_tails(lsf, threshold_ratio=0.05, fit_points=5)

    # Compute FFT and frequencies
    otf_1d = np.fft.fft(np.fft.ifftshift(lsf))
    mtf_1d = np.abs(otf_1d)
    mtf_1d = (
        mtf_1d / mtf_1d[0] if mtf_1d[0] != 0 else mtf_1d / (mtf_1d[0] + 1e-10)
    )  # Normalize to 1 at zero frequency, avoiding division by zero

    freq = np.fft.fftfreq(lsf.size, d=pixel_size)

    # Shift for plotting
    mtf_1d = np.fft.fftshift(mtf_1d)
    freq = np.fft.fftshift(freq)

    # Consider only positive frequencies
    mask = freq >= 0
    freq_pos = freq[mask]
    mtf_pos = mtf_1d[mask]

    # Find MTF10 (interpolating if needed)
    mtf10_value = 0.10
    if np.any(mtf_pos <= mtf10_value):
        idx = np.where(mtf_pos <= mtf10_value)[0][0]
        # Linear interpolation
        if idx == 0:
            mtf10_freq = freq_pos[0]
        else:
            f1, f2 = freq_pos[idx - 1], freq_pos[idx]
            m1, m2 = mtf_pos[idx - 1], mtf_pos[idx]
            mtf10_freq = f1 + (mtf10_value - m1) * (f2 - f1) / (m2 - m1)
    else:
        mtf10_freq = np.nan  # Not reached

    return freq_pos, mtf_pos, mtf10_freq


def get_mtf_at_freq(
    target_freq: float, freq_array: np.ndarray, mtf_array: np.ndarray
) -> float:
    """
    Finds the MTF value at a specific frequency using linear interpolation.

    Parameters
    ----------
    target_freq
        The frequency on which to evaluate the MTF.
    freq_array
        The array of frequency values.
    mtf_array
        The array of MTF values.

    Returns
    -------
    float
        The interpolated MTF value at the target frequency.
    """
    return np.interp(target_freq, freq_array, mtf_array)


if __name__ == "__main__":
    from imageio import imread
    from .plotters import plot_1d_mtf

    psf_path = r"C:\Users\jacop\Desktop\PhD\Focal Spot\Input images\virtual_images\psf\PSF-downsampled.png"
    pixel_size = 0.154  # mm

    psf = imread(
        psf_path,
    )
    psf = psf / np.sum(psf)
    freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size)
    print(f"MTF10 frequency: {mtf10} cycles/mm")
    plot_1d_mtf(
        freq,
        mtf_1d,
        pixel_size,
        out_path="mtf_1d_example.png",
        mtf10_freq=mtf10,
        show_plots=True,
    )
