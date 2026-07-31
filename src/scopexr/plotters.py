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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_circle_on_crop(
    cropped: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    output_path: str | Path,
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
    plt.savefig(Path(output_path) / "circle_on_crop.png", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_profiles_and_reconstruction(
    profiles: np.ndarray,
    sinogram: np.ndarray,
    reconstruction: np.ndarray,
    out_dir: str | Path,
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
    plt.xlabel("Angle Index")
    plt.ylabel("Radial Offset (px)")

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
    plt.savefig(Path(out_dir) / f"profiles_sinogram_reconstruction{suffix}.png")
    if show_plots:
        plt.show()
    plt.close()


def plot_profiles_with_fwhm(
    radial: np.ndarray,
    prof_horizontal_sino: np.ndarray,
    prof_vertical_sino: np.ndarray,
    horizontal_idx: int,
    vertical_idx: int,
    height: float,
    fh: float,
    lh: float,
    rh: float,
    fv: float,
    lv: float,
    rv: float,
    out_path: str | Path,
    show_plots: bool = False,
    pixel_size: float|None = None,
    magnification: float|None = None,
) -> None:
    """
    Plot the horizontal and vertical sinogram profiles with FWHM/FW15M lines.

    Parameters
    ----------
    radial
        Radial coordinates array (in pixels).
    prof_horizontal_sino
        Intensity array of the horizontal profile.
    prof_vertical_sino
        Intensity array of the vertical profile.
    horizontal_idx
        Index of the horizontal profile.
    vertical_idx
        Index of the vertical profile.
    height
        Relative height for the width measurement (e.g., 0.5 for FWHM).
    fh
        Width of the horizontal profile (in pixels).
    lh
        Left coordinate of the horizontal profile width.
    rh
        Right coordinate of the horizontal profile width.
    fv
        Width of the vertical profile (in pixels).
    lv
        Left coordinate of the vertical profile width.
    rv
        Right coordinate of the vertical profile width.
    out_path
        Path to save the figure.
    show_plots
        If True, display the plot.
    pixel_size
        Pixel size in mm. If provided with magnification, x-axis will be in mm.
    magnification
        Magnification factor. If provided with pixel_size, x-axis will be in mm.

    Returns
    -------
    None
        This function saves a file and does not return a value.
    """
    # Convert to mm if pixel_size and magnification are provided
    if pixel_size is not None and magnification is not None:
        scale_factor = pixel_size / magnification
        radial_display = radial * scale_factor
        fh_display = fh * scale_factor
        fv_display = fv * scale_factor
        x_label = "Radial Offset (mm)"
        width_unit = "mm"
    else:
        radial_display = radial
        fh_display = fh
        fv_display = fv
        x_label = "Radial Offset (pixels)"
        width_unit = "px"

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the two sinogram profiles
    ax.plot(
        radial_display,
        prof_horizontal_sino,
        label=f"Horizontal (idx={horizontal_idx})",
        color="teal",
    )
    ax.plot(
        radial_display,
        prof_vertical_sino,
        label=f"Vertical (idx={vertical_idx})",
        color="orange",
    )

    # Compute threshold levels (profiles are baseline-subtracted, so use peak * height)
    half_h = prof_horizontal_sino.max() * height
    half_v = prof_vertical_sino.max() * height

    # Interpolate radial coordinates at fractional indices
    idx = np.arange(len(radial))
    lh_val = np.interp(lh, idx, radial_display)
    rh_val = np.interp(rh, idx, radial_display)
    lv_val = np.interp(lv, idx, radial_display)
    rv_val = np.interp(rv, idx, radial_display)

    # Draw the half-max horizontal lines spanning between left/right edges
    ax.hlines(
        half_h,
        lh_val,
        rh_val,
        linestyles="-.",
        color="teal",
        label=f"Horizontal FWHM = {fh_display:.2f}{width_unit}",
    )
    ax.hlines(
        half_v,
        lv_val,
        rv_val,
        linestyles="--",
        color="orange",
        label=f"Vertical FWHM = {fv_display:.2f}{width_unit}",
    )

    ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

    ax.set_xlabel(x_label)
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
    horizontal_idx: int,
    vertical_idx: int,
    out_path: str | Path,
    reconstruction_type: str,
    show_plots: bool,
) -> None:
    """
    Plot the sinogram with vertical lines indicating the selected profiles.

    Parameters
    ----------
    sinogram
        The sinogram image/array.
    horizontal_idx
        Index of the horizontal profile.
    vertical_idx
        Index of the vertical profile.
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
        ax.set_title("Sinogram with Horizontal and Vertical Profiles")
        ax.axvline(
            horizontal_idx,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Horizontal (idx={horizontal_idx})",
        )
        ax.axvline(
            vertical_idx,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Vertical (idx={vertical_idx})",
        )
    else:
        ax.set_title("Sinogram with Horizontal and Vertical Profiles")
        ax.axvline(
            horizontal_idx,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Horizontal (idx={horizontal_idx})",
        )
        ax.axvline(
            vertical_idx,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Vertical (idx={vertical_idx})",
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
    angle_horizontal: float,
    angle_vertical: float,
    out_path: str | Path,
    show_plots: bool = False,
    reconstruction_type: str = "fs",
) -> None:
    """
    Plot the reconstruction with lines indicating the profile angles.

    Parameters
    ----------
    recon
        2D reconstruction image.
    angle_horizontal
        Angle in degrees for the horizontal profile.
    angle_vertical
        Angle in degrees for the vertical profile.
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
    ax.imshow(img, cmap="gray", extent=(0, w, 0, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")

    half_diag = int(np.sqrt(h**2 + w**2) / 2) + 10
    for angle, color in [(angle_horizontal, "red"), (angle_vertical, "blue")]:
        theta = np.deg2rad(angle)
        dx = half_diag * np.cos(theta)
        dy = half_diag * np.sin(theta)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, linewidth=2)

    if reconstruction_type == "psf":
        ax.set_title("PSF with Horizontal and Vertical Profiles")
        legend_labels = ["Horizontal (0°)", "Vertical (90°)"]
    else:
        ax.set_title("Focal Spot with Horizontal and Vertical Profiles")
        legend_labels = [
            f"Horizontal (angle={angle_horizontal}°)",
            f"Vertical (angle={angle_vertical}°)",
        ]

    ax.legend(legend_labels)
    ax.axis("off")

    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_profile_with_gaussian(
    radial: np.ndarray,
    intensity_profile: np.ndarray,
    popt: tuple[float, float, float, float],
    out_path: str | Path,
    show_plots: bool = False,
    pixel_size: float | None = None,
    magnification: float | None = None,
) -> None:
    """
    Plot an intensity profile (from sinogram or projection) with its Gaussian fit.
    """
    A, mu, sigma, B = popt

    # Convert to mm if pixel_size and magnification are provided
    if pixel_size is not None and magnification is not None:
        scale_factor = pixel_size / magnification
        radial_display = radial * scale_factor
        x_label = "Radial Position (mm)"

        mu_display = mu * scale_factor
        sigma_display = sigma * scale_factor
    else:
        radial_display = radial
        x_label = "Radial Position (px)"
        mu_display = mu
        sigma_display = sigma

    # Create a dense index axis for smooth curve
    radial_dense = np.linspace(radial_display[0], radial_display[-1], 500)

    # Compute fitted Gaussian using the properly scaled parameters
    fitted_dense = (
        A * np.exp(-((radial_dense - mu_display) ** 2) / (2 * sigma_display**2)) + B
    )

    plt.figure(figsize=(8, 4))
    plt.plot(radial_display, intensity_profile, label="Data")
    plt.plot(radial_dense, fitted_dense, linestyle="--", label="Gaussian Fit")

    plt.title("Intensity Profile with Gaussian Fit")
    plt.xlabel(x_label)
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
    out_path: str | Path,
    mtf10_freq: float | None = None,
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
