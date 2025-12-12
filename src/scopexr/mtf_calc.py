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

def compute_1d_mtf(psf: np.ndarray, pixel_size: float, axis: int)-> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute 1D MTF from 2D PSF by integrating to LSF along specified axis.

    Parameters
    ----------
    psf
        2D array representing the point spread function (PSF).
    pixel_size
        Pixel size in mm.
    axis
        Axis along which to integrate (0 for rows, 1 for columns).

    Returns
    -------
    freq: np.ndarray
        Frequencies in cycles/mm.
    mtf_1d: np.ndarray
        1D MTF array normalized to 1 at zero frequency.
    mtf10: float
        Frequency at which MTF drops to 10% (cycles/mm).
    """
    # Compute LSF
    lsf = np.sum(psf, axis=axis)
    lsf = lsf / np.sum(lsf)

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


def compute_1d_mtf_from_sino(sinogram: np.ndarray, pixel_size: float, angle: int)-> tuple[np.ndarray, np.ndarray, float]:
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

    lsf = lsf / np.sum(lsf)

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
            # Avoid division by zero if m2 == m1
            if m2 == m1:
                mtf10_freq = f1
            else:
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
    psf_path = r"C:\Users\jacop\Desktop\PhD\Focal Spot\Input images\virtual_images\psf\PSF-downsampled.png"
    from imageio import imread
    from .plotters import plot_1d_mtf

    psf = imread(
        psf_path,
    )
    psf = psf / np.sum(psf)
    pixel_size = 0.154  # mm
    freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size, axis=0)
    print(f"MTF10 frequency: {mtf10} cycles/mm")
    plot_1d_mtf(
        freq,
        mtf_1d,
        pixel_size,
        out_path="mtf_1d_example.png",
        mtf10_freq=mtf10,
        show_plots=True,
    )
