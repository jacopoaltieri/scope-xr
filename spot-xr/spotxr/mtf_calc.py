import numpy as np
import matplotlib.pyplot as plt


# def compute_2d_mtf(psf):
#     """
#     Compute 2D MTF from 2D PSF.
#     """
#     psf = psf / np.sum(psf)  # Normalize PSF
#     otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
#     mtf = np.abs(otf)
#     mtf = mtf / np.max(mtf)  # Normalize MTF to 1 at zero frequency
#     return mtf


# def plot_2d_mtf(mtf, title="2D MTF"):
#     """
#     Plot 2D MTF.
#     """
#     plt.figure(figsize=(6, 5))
#     plt.imshow(mtf, cmap="jet", extent=[-0.5, 0.5, -0.5, 0.5])
#     plt.title(title)
#     plt.xlabel("Spatial frequency fx (cycles/pixel)")
#     plt.ylabel("Spatial frequency fy (cycles/pixel)")
#     plt.colorbar(label="MTF")
#     plt.show()


def compute_1d_mtf(psf: np.ndarray, pixel_size: float, axis: int):
    """
    Compute 1D MTF from 2D PSF by integrating to LSF along specified axis.

    Args:
        psf: 2D array representing the point spread function (PSF).
        pixel_size: Pixel size in mm.
        axis: Axis along which to integrate (0 for rows, 1 for columns).

    Returns:
        freq: Frequencies in cycles/mm.
        mtf_1d: 1D MTF array normalized to 1 at zero frequency.
        mtf10: Frequency at which MTF drops to 10% (cycles/mm).
    """
    # Compute LSF
    lsf = np.sum(psf, axis=axis)
    lsf = lsf / np.sum(lsf)

    # Compute FFT and frequencies
    otf_1d = np.fft.fft(lsf)
    mtf_1d = np.abs(otf_1d)
    mtf_1d = mtf_1d / np.max(mtf_1d)

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

    # Compute MTF at Nyquist frequency
    nyquist_freq = 1 / (2 * pixel_size)

    if nyquist_freq <= freq_pos[-1]:
        # Interpolate
        idx = np.searchsorted(freq_pos, nyquist_freq)
        if idx == 0:
            mtf_nyquist = mtf_pos[0]
        else:
            f1, f2 = freq_pos[idx - 1], freq_pos[idx]
            m1, m2 = mtf_pos[idx - 1], mtf_pos[idx]
            mtf_nyquist = m1 + (nyquist_freq - f1) * (m2 - m1) / (f2 - f1)
    else:
        mtf_nyquist = np.nan  # Beyond available frequency range

    return freq_pos, mtf_pos, mtf10_freq, mtf_nyquist
