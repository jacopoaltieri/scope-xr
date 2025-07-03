import numpy as np
import matplotlib.pyplot as plt

def compute_2d_mtf(psf):
    """
    Compute 2D MTF from 2D PSF.
    """
    psf = psf / np.sum(psf)  # Normalize PSF
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    mtf = np.abs(otf)
    mtf = mtf / np.max(mtf)  # Normalize MTF to 1 at zero frequency
    return mtf

def plot_2d_mtf(mtf, title='2D MTF'):
    """
    Plot 2D MTF.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(mtf, cmap='jet', extent=[-0.5, 0.5, -0.5, 0.5])
    plt.title(title)
    plt.xlabel('Spatial frequency fx (cycles/pixel)')
    plt.ylabel('Spatial frequency fy (cycles/pixel)')
    plt.colorbar(label='MTF')
    plt.show()


def compute_1d_mtf_from_lsf(psf, axis=0):
    """
    Compute 1D MTF from 2D PSF by integrating to LSF along specified axis.
    
    axis = 0 → integrate along rows (get horizontal LSF)
    axis = 1 → integrate along columns (get vertical LSF)
    """
    psf = psf / np.sum(psf)  # Normalize
    lsf = np.sum(psf, axis=axis)
    lsf = lsf / np.sum(lsf)  # Normalize LSF
    otf_1d = np.fft.fftshift(np.fft.fft(lsf))
    mtf_1d = np.abs(otf_1d)
    mtf_1d = mtf_1d / np.max(mtf_1d)  # Normalize
    return mtf_1d

def plot_1d_mtf(mtf_1d, pixel_size_mm, title='1D MTF'):
    """
    Plot 1D MTF vs spatial frequency in cycles/mm.
    Shows only positive frequencies and a Nyquist line.
    
    pixel_size_mm: pixel pitch in mm (e.g., 0.005 for 5 μm)
    """
    n = len(mtf_1d)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=pixel_size_mm))  # in cycles/mm
    
    # Only positive frequencies
    half = n // 2
    freqs_pos = freqs[half:]
    mtf_pos = mtf_1d[half:]
    
    # Nyquist frequency
    f_nyquist = 1 / (2 * pixel_size_mm)
    
    plt.figure(figsize=(6, 4))
    plt.plot(freqs_pos, mtf_pos, label='MTF')
    plt.axvline(f_nyquist, color='r', linestyle='--', label=f'Nyquist ({f_nyquist:.2f} cy/mm)')
    plt.title(title)
    plt.xlabel('Spatial frequency (cycles/mm)')
    plt.ylabel('MTF')
    plt.grid(True)
    plt.legend()
    plt.xlim([0, freqs_pos[-1]])
    plt.ylim([0, 1.05])
    plt.show()
