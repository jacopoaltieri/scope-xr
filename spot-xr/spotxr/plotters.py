import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_circle_on_crop(
    cropped: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    output_path: str,
    show: bool = False,
) -> None:

    fig, ax = plt.subplots()
    ax.imshow(cropped, cmap="gray")

    # Draw circle
    ax.add_patch(Circle((cx, cy), radius, edgecolor="red", fill=False, linewidth=2))

    # Draw center
    ax.plot(cx, cy, "ro", markersize=5)

    ax.set_title("Estimated Radius with Center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path + "/circle_on_crop.png", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_profiles_and_reconstruction(
    profiles, sinogram, reconstruction, out_dir, show_plots, suffix=""
):
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
    plt.title("Reconstructed Focal Spot")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"profiles_sinogram_reconstruction{suffix}.png"))
    if show_plots:
        plt.show()
    plt.close()


def plot_profiles_with_fwhm(
    radial,
    prof_wide_sino,
    prof_narrow_sino,
    wide_idx,
    narrow_idx,
    fw,
    lw,
    rw,
    fn,
    ln,
    rn,
    out_path,
    show_plots=False,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the two sinogram profiles
    ax.plot(radial, prof_wide_sino, label=f"Widest (idx={wide_idx})")
    ax.plot(radial, prof_narrow_sino, label=f"Narrowest (idx={narrow_idx})")

    # Compute half-max levels
    half_w = (prof_wide_sino.max() + prof_wide_sino.min()) / 2.0
    half_n = (prof_narrow_sino.max() + prof_narrow_sino.min()) / 2.0

    # Draw the half-max horizontal lines spanning between left/right edges
    ax.hlines(
        half_w,
        radial[lw],
        radial[rw],
        linestyles="--",
        color="red",
        label=f"Widest FWHM = {fw}px",
    )
    ax.hlines(
        half_n,
        radial[ln],
        radial[rn],
        linestyles="--",
        color="red",
        label=f"Narrowest FWHM = {fn}px",
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
    sinogram, wide_idx, narrow_idx, out_path, show_plots=False
):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sinogram, cmap="gray", aspect="auto")
    ax.set_title("Sinogram with Widest & Narrowest Profiles")
    ax.set_xlabel("Angle Index")
    ax.set_ylabel("Radial Offset (px)")

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

    ax.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_recon_with_lines(recon, angle_wide, angle_narrow, out_path, show_plots=False):
    """
    recon: 2D np.ndarray image
    center: (cx, cy) tuple - center of the focal spot
    angle_wide, angle_narrow: angles in degrees where lines should be drawn
    """

    img = recon.copy()
    w, h = img.shape[:2]
    cx = w / 2
    cy = h / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    # show only the image region
    ax.imshow(img, cmap="gray", extent=[0, w, 0, h])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")

    # now draw the lines
    half_diag = int(np.sqrt(h**2 + w**2) / 2) + 10
    for angle, color in [(angle_wide, "red"), (angle_narrow, "blue")]:
        theta = np.deg2rad(angle)
        dx = half_diag * np.cos(theta)
        dy = half_diag * np.sin(theta)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, linewidth=2)

    ax.set_title("Focal Spot with Widest & Narrowest Profiles")
    ax.legend([f"Widest (angle={angle_wide})", f"Narrowest (angle={angle_narrow})"])
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
):
    """
    Plot a sinogram profile with its Gaussian fit.

    Args:
        radial: 1D array of radial positions (centered, e.g. -L..+L).
        sinogram_profile: 1D array of intensity values.
        popt: Optimal parameters from Gaussian fit [A, mu, sigma, B]
              where mu is in index space (0..n-1).
        out_path: Path to save the plot.
        show_plots: Whether to display the plot interactively.
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

    # Map those dense indices back onto the radial axis

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
    freq, mtf, pixel_size, out_path, mtf10_freq=None, mtf_nyquist=None, show_plots=False
):
    """
    Plot 1D MTF with Nyquist and MTF10 reference lines.

    Args:
        freq: Array of frequencies in cycles/mm.
        mtf: MTF values (same length as freq).
        pixel_size: Pixel size in mm (system pixel size!).
        mtf10_freq: Frequency at which MTF drops to 10% (cycles/mm).
        mtf_nyquist: MTF value at Nyquist frequency (optional).
        out_path: Path to save the figure.
        show_plots: If True, also display plot on screen.
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

    # MTF value at Nyquist marker
    if mtf_nyquist is not None and not np.isnan(mtf_nyquist):
        plt.plot(
            nyquist_freq, mtf_nyquist, "ro", label=f"MTF@Nyquist = {mtf_nyquist:.2f}"
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
    plt.xlim([0, nyquist_freq * 1.1])
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()
