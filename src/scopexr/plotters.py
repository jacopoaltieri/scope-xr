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

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle


def plot_circle_on_crop(
    cropped: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    output_path: str,
    show: bool = False,
) -> None:
    """
    Plot the cropped ROI with the detected circle overlay.

    Parameters
    ----------
    cropped
        The 2D image array of the crop.
    cx
        Center x-coordinate relative to the crop.
    cy
        Center y-coordinate relative to the crop.
    radius
        Radius of the circle in pixels.
    output_path
        Directory to save the image.
    show
        If True, display the plot.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    fig, ax = plt.subplots()
    ax.imshow(cropped, cmap="gray")

    # Draw circle
    ax.add_patch(Circle((cx, cy), radius, edgecolor="red", fill=False, linewidth=2))

    # Draw center
    ax.plot(cx, cy, "ro", markersize=5)

    ax.set_title("Cropped ROI around circle")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "circle_on_crop.png"), dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_profiles_and_reconstruction(
    profiles: np.ndarray,
    sinogram: np.ndarray,
    reconstruction: np.ndarray,
    out_dir: str,
    show_plots: bool,
    reconstruction_type: str,
    suffix: str = "",
) -> None:
    """
    Plot aligned profiles, sinogram, and reconstruction side-by-side.

    Parameters
    ----------
    profiles
        The aligned profiles image/array.
    sinogram
        The sinogram image/array.
    reconstruction
        The reconstructed image/array.
    out_dir
        Directory to save the plot.
    show_plots
        If True, display the plot.
    reconstruction_type
        Type string ('psf', 'fs', or other) to determine the title.
    suffix
        Optional suffix for the output filename.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(profiles, cmap="gray")
    plt.title("Aligned Profiles")
    plt.xlabel("Profile Index")
    plt.ylabel("Angle Index")

    plt.subplot(1, 3, 2)
    plt.imshow(sinogram, cmap="gray")
    plt.title("Sinogram")
    plt.xlabel("Angle Index")
    plt.ylabel("Radial Offset (px)")

    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction, cmap="gray")
    if reconstruction_type == "psf":
        title = "Reconstructed PSF"
    elif reconstruction_type == "fs":
        title = "Reconstructed Focal Spot"
    else:
        title = "Reconstruction"
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"profiles_sinogram_reconstruction{suffix}.png"))
    if show_plots:
        plt.show()
    plt.close()


def plot_profiles_with_fwhm(
    radial: np.ndarray,
    prof_wide_sino: np.ndarray,
    prof_narrow_sino: np.ndarray,
    wide_idx: int,
    narrow_idx: int,
    height: float,
    fw: float,
    lw: float,
    rw: float,
    fn: float,
    ln: float,
    rn: float,
    out_path: str,
    show_plots: bool = False,
) -> None:
    """
    Plot the widest and narrowest sinogram profiles with FWHM/FW10M lines.

    Parameters
    ----------
    radial
        Radial coordinates array.
    prof_wide_sino
        Intensity array of the widest profile.
    prof_narrow_sino
        Intensity array of the narrowest profile.
    wide_idx
        Index of the widest profile.
    narrow_idx
        Index of the narrowest profile.
    height
        Relative height for the width measurement (e.g., 0.5 for FWHM).
    fw
        Width of the widest profile.
    lw
        Left coordinate of the widest profile width.
    rw
        Right coordinate of the widest profile width.
    fn
        Width of the narrowest profile.
    ln
        Left coordinate of the narrowest profile width.
    rn
        Right coordinate of the narrowest profile width.
    out_path
        Path to save the figure.
    show_plots
        If True, display the plot.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the two sinogram profiles
    ax.plot(radial, prof_wide_sino, label=f"Widest (idx={wide_idx})", color="teal")
    ax.plot(
        radial, prof_narrow_sino, label=f"Narrowest (idx={narrow_idx})", color="orange"
    )

    # Compute half-max levels
    half_w = (
        prof_wide_sino.max() - prof_wide_sino.min()
    ) * height + prof_wide_sino.min()
    half_n = (
        prof_narrow_sino.max() - prof_narrow_sino.min()
    ) * height + prof_narrow_sino.min()

    # Interpolate radial coordinates at fractional indices
    idx = np.arange(len(radial))
    lw_val = np.interp(lw, idx, radial)
    rw_val = np.interp(rw, idx, radial)
    ln_val = np.interp(ln, idx, radial)
    rn_val = np.interp(rn, idx, radial)

    # Draw the half-max horizontal lines spanning between left/right edges
    ax.hlines(
        half_w,
        lw_val,
        rw_val,
        linestyles="-.",
        color="teal",
        label=f"Widest FWHM = {fw:.2f}px",
    )
    ax.hlines(
        half_n,
        ln_val,
        rn_val,
        linestyles="--",
        color="orange",
        label=f"Narrowest FWHM = {fn:.2f}px",
    )

    ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

    ax.set_xlabel("Radial Offset (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title("Central FWHM on Sinogram Profiles")
    ax.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_sinogram_with_traced_profiles(
    sinogram: np.ndarray,
    wide_idx: int,
    narrow_idx: int,
    out_path: str,
    reconstruction_type: str,
    show_plots: bool,
) -> None:
    """
    Plot the sinogram with vertical lines indicating the selected profiles.

    Parameters
    ----------
    sinogram
        The sinogram image/array.
    wide_idx
        Index of the widest (or horizontal) profile.
    narrow_idx
        Index of the narrowest (or vertical) profile.
    out_path
        Path to save the figure.
    reconstruction_type
        'psf' or 'fs' to determine label text.
    show_plots
        If True, display the plot.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sinogram, cmap="gray", aspect="auto")

    if reconstruction_type == "psf":
        ax.set_title("Sinogram with Horizontal & Vertical Profiles")
        ax.axvline(
            wide_idx,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Horizontal (idx={wide_idx})",
        )
        ax.axvline(
            narrow_idx,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Vertical (idx={narrow_idx})",
        )
    else:
        ax.set_title("Sinogram with Widest & Narrowest Profiles")
        ax.axvline(
            wide_idx,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Widest (idx={wide_idx})",
        )
        ax.axvline(
            narrow_idx,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Narrowest (idx={narrow_idx})",
        )

    ax.set_xlabel("Angle Index")
    ax.set_ylabel("Radial Offset (px)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_recon_with_lines(
    recon: np.ndarray,
    angle_wide: float,
    angle_narrow: float,
    out_path: str,
    show_plots: bool = False,
    reconstruction_type: str = "fs",
) -> None:
    """
    Plot the reconstruction with lines indicating the profile angles.

    Parameters
    ----------
    recon
        2D reconstruction image.
    angle_wide
        Angle in degrees for the widest profile.
    angle_narrow
        Angle in degrees for the narrowest profile.
    out_path
        Path to save the figure.
    show_plots
        If True, display the plot.
    reconstruction_type
        'psf' or 'fs' to determine title and labels.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    img = recon.copy()
    w, h = img.shape[:2]
    cx = w / 2
    cy = h / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", extent=[0, w, 0, h])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")

    half_diag = int(np.sqrt(h**2 + w**2) / 2) + 10
    for angle, color in [(angle_wide, "red"), (angle_narrow, "blue")]:
        theta = np.deg2rad(angle)
        dx = half_diag * np.cos(theta)
        dy = half_diag * np.sin(theta)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, linewidth=2)

    if reconstruction_type == "psf":
        ax.set_title("PSF with Horizontal & Vertical Profiles")
        legend_labels = [f"Horizontal (0°)", f"Vertical (90°)"]
    else:
        ax.set_title("Focal Spot with Widest & Narrowest Profiles")
        legend_labels = [
            f"Widest (angle={angle_wide}°)",
            f"Narrowest (angle={angle_narrow}°)",
        ]

    ax.legend(legend_labels)
    ax.axis("off")

    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_profile_with_gaussian(
    radial: np.ndarray,
    sinogram_profile: np.ndarray,
    popt: tuple[float, float, float, float],
    out_path: str,
    show_plots: bool = False,
) -> None:
    """
    Plot a sinogram profile with its Gaussian fit.

    Parameters
    ----------
    radial
        1D array of radial positions (centered, e.g. -L..+L).
    sinogram_profile
        1D array of intensity values.
    popt
        Optimal parameters from Gaussian fit [A, mu, sigma, B]
        where mu is in index space (0..n-1).
    out_path
        Path to save the plot.
    show_plots
        Whether to display the plot interactively.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    n = sinogram_profile.size
    center = n // 2
    spacing = radial[1] - radial[0]

    A, mu, sigma, B = popt
    mu_phys = (mu - center) * spacing
    sigma_phys = sigma * spacing

    # Create a dense index axis for smooth curve
    radial_dense = np.linspace(radial[0], radial[-1], 500)
    # Compute fitted Gaussian in index‐space
    fitted_dense = (
        A * np.exp(-((radial_dense - mu_phys) ** 2) / (2 * sigma_phys**2)) + B
    )

    plt.figure(figsize=(8, 4))
    plt.plot(radial, sinogram_profile, label="Data")
    plt.plot(radial_dense, fitted_dense, linestyle="--", label=f"Gaussian Fit")

    plt.title("Sinogram Profile with Gaussian Fit")
    plt.xlabel("Radial Position (px)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()


def plot_1d_mtf(
    freq: np.ndarray,
    mtf: np.ndarray,
    pixel_size: float,
    out_path: str,
    mtf10_freq: float = None,
    show_plots: bool = False,
) -> None:
    """
    Plot 1D MTF with Nyquist and MTF10 reference lines.

    Parameters
    ----------
    freq
        Array of frequencies in cycles/mm.
    mtf
        MTF values (same length as freq).
    pixel_size
        Pixel size in mm (system pixel size!).
    out_path
        Path to save the figure.
    mtf10_freq
        Frequency at which MTF drops to 10% (cycles/mm).
    show_plots
        If True, also display plot on screen.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    # Nyquist frequency in cycles/mm
    nyquist_freq = 1 / (2 * pixel_size)

    plt.figure(figsize=(8, 5))
    plt.plot(freq, mtf, label="MTF curve", lw=2)

    # Vertical line at Nyquist
    plt.axvline(
        nyquist_freq,
        color="r",
        linestyle="--",
        label=f"Nyquist = {nyquist_freq:.2f} cy/mm",
    )

    # Horizontal line at 10% until MTF10
    if mtf10_freq is not None and not np.isnan(mtf10_freq):
        plt.hlines(
            0.1,
            0,
            mtf10_freq,
            colors="gray",
            linestyles=":",
            label=f"MTF10 = {mtf10_freq:.2f} cy/mm",
        )

    plt.xlabel("Spatial frequency [cycles/mm]")
    plt.ylabel("MTF")
    plt.title("1D Modulation Transfer Function (MTF)")
    plt.ylim([0, 1.05])
    plt.xlim([0, nyquist_freq * 1.05])
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()
