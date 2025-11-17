import numpy as np
import matplotlib.pyplot as plt
from scopexr.mtf_calc import compute_1d_mtf
import scopexr.image_opening as io
import scopexr.circle_detection as circ
import scopexr.sinogram_recon as sr
from scipy.stats import binned_statistic

def compute_subpixel_profiles_and_sinogram_traditional(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int,
    profile_half_length: int,
    derivative_step: int,
    dtheta: float,
    resample_radial: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes sub-pixel edge profiles and sinogram by interpolating in angular wedges.

    Args:
        img: 2D grayscale image array.
        cx: X-coordinate of the circle center.
        cy: Y-coordinate of the circle center.
        radius: Radius of the circle.
        n_angles: Number of angular samples (in full 360°).
        profile_half_length: Half-length (in pixels) of radial sampling.
        derivative_step: Step size for derivative computation.
        dtheta: Angular width (degrees) of wedge around each angle.
        resample_radial: Radial step for interpolation grid (in pixels).

    Returns:
        profiles: 2D array of shape (profile_bins, n_angles), radial profiles.
        sinogram: 2D array of shape (profile_bins, n_angles), negative radial derivatives.
    """
    # Assuming sr._check_phl is available from scopexr.sinogram_recon
    profile_half_length = sr._check_phl(img, cx, radius, profile_half_length)

    # Convert angles and angular wedge width to radians
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    half_wedge = np.deg2rad(dtheta) / 2

    # Coordinates relative to center
    ys, xs = np.indices(img.shape)
    xs = xs.astype(np.float32) - cx
    ys = ys.astype(np.float32) - cy
    phis = np.arctan2(ys, xs)
    rs = np.hypot(xs, ys) - radius

    # Set up radial grid for interpolation
    min_r = -profile_half_length
    max_r = profile_half_length
    r_grid = np.arange(min_r, max_r + 1 / resample_radial, 1 / resample_radial)
    n_bins = r_grid.size

    # Initialize profiles array (angles x radial positions)
    profiles = np.full((n_angles, n_bins), np.nan, dtype=np.float32)

    for i, theta in enumerate(angles):
        # mask pixels in angular wedge
        dphi = (phis - theta + np.pi) % (2 * np.pi) - np.pi
        mask = np.abs(dphi) <= half_wedge
        r_vals = rs[mask]
        intensities = img[mask]

        # Restrict to radial range
        radial_mask = (r_vals >= min_r) & (r_vals <= max_r)
        r_vals = r_vals[radial_mask]
        intensities = intensities[radial_mask]

        # Sort for interpolation
        idx = np.argsort(r_vals)
        r_vals = r_vals[idx]
        intensities = intensities[idx]
        bin_edges = np.append(r_grid, r_grid[-1] + (1/resample_radial))
        
        # Interpolate to uniform grid
        # Handle empty wedge case to avoid interp error
        if r_vals.size > 0:
            # 'statistic="mean"': averages all intensities in each bin
            # 'bins=bin_edges': uses your high-res grid as the bins
            # 'x=r_vals': the positions of the blue dots
            # 'values=intensities': the values of the blue dots
            bin_means, _, _ = binned_statistic(
                r_vals,
                intensities,
                statistic='mean',
                bins=bin_edges
            )
            
            # bin_means will have NaN for any bin that had 0 points.
            # We must fill these small gaps.
            nan_mask = np.isnan(bin_means)
            if np.any(nan_mask) and not np.all(nan_mask):
                # Create an x-coordinate array for interpolation
                x = np.arange(bin_means.size)
                # Interpolate ONLY the nan values
                bin_means[nan_mask] = np.interp(
                    x[nan_mask], # points to interpolate
                    x[~nan_mask], # known x's
                    bin_means[~nan_mask] # known y's
                )
            interp_vals = bin_means
        else:
            interp_vals = np.full(r_grid.shape, np.nan)
        # --- END NEW METHOD ---

        # old line:
        # interp_vals = np.interp(r_grid, r_vals, intensities)
        profiles[i, :] = interp_vals

    # Compute radial derivative
    sinogram = np.gradient(profiles, derivative_step, axis=1)

    return profiles.T, -sinogram.T


def compute_mtf_from_sinogram(sinogram: np.ndarray, pixel_size: float, angle: int):
    """
    Compute 1D MTF from sinogram

    Args:
        sinogram: 2D array representing the sinogram.
        pixel_size: Pixel size in mm.
        angle: Angle index along which to extract the profile.
    Returns:
        freq: Frequencies in cycles/mm.
        mtf_1d: 1D MTF array normalized to 1 at zero frequency.
        mtf10: Frequency at which MTF drops to 10% (cycles/mm).
    """
    lsf = sinogram[:, angle]
    
    # --- Baseline Subtraction ---
    # Fix for loaded TIFFs that may not be zero-based
    baseline = np.nanmin(lsf) # Use nanmin to ignore NaNs
    lsf = lsf - baseline

    # Handle NaNs from empty wedges if any (e.g., by interpolation or setting to 0)
    nan_mask = np.isnan(lsf)
    if np.any(nan_mask):
        lsf[nan_mask] = 0.0 # Simple: set NaNs to 0
        
    if np.sum(lsf) == 0: # Avoid division by zero if profile is all 0/NaN
        return np.nan, np.nan, np.nan

    lsf = lsf / np.sum(lsf)

    # Compute FFT and frequencies
    otf_1d = np.fft.fft(np.fft.ifftshift(lsf))
    mtf_1d = np.abs(otf_1d)
    
    if mtf_1d[0] == 0: # Avoid division by zero
        return np.nan, np.nan, np.nan
        
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


def plot_1d_mtf(freq, mtf, pixel_size, out_path, mtf10_freq=None, show_plots=False):
    """
    Plot 1D MTF with Nyquist and MTF10 reference lines.

    Args:
        freq: Array of frequencies in cycles/mm.
        mtf: MTF values (same length as freq).
        pixel_size: Pixel size in mm (system pixel size!).
        mtf10_freq: Frequency at which MTF drops to 10% (cycles/mm).
        out_path: Path to save the figure.
        show_plots: If True, also display plot on screen.
    """
    # Handle NaN inputs gracefully
    if freq is np.nan or mtf is np.nan:
        print(f"Skipping plot {out_path} due to NaN data.")
        return

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
    # Adjust xlim to Nyquist, or max freq if Nyquist is too small
    plt.xlim([0, max(nyquist_freq * 1.05, np.nanmax(freq) * 0.1)])
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300) # Save the plot
    if show_plots:
        plt.show()
    plt.close() # Close the figure


if __name__ == "__main__":
    pixel_size = 0.050  # mm
    n_angles = 360
    profile_half_length = 50
    derivative_step = 1
    filter_name = "ramp"
    symmetrize = False
    dtheta = 2
    resample2 = 4

    input_path = r"C:\Users\jacop\Desktop\PhD\Focal Spot\Input images\AVG_cerchio_2_70kvp_100uA_1mmAl_500ms_LFW_2802x2400x10_corr.tif"

    img = io.load_image(input_path)
    # ... (normal sino/recon calculation) ...
    cx, cy, radius = circ.estimate_circle(img)
    profiles, sinogram = sr.compute_profiles_and_sinogram(
        img, cx, cy, radius, n_angles, profile_half_length, derivative_step
    )
    centered_sino, applied_shift = sr.auto_center_sinogram(sinogram)
    sinogram = centered_sino
    reconstruction = sr.reconstruct_focal_spot(sinogram, filter_name, symmetrize)


    # --- This call now uses the new Bin+Fill+Smooth function ---
    sub_profiles, sub_sinogram = compute_subpixel_profiles_and_sinogram_traditional(
        img,
        cx,
        cy,
        radius,
        n_angles,
        profile_half_length,
        derivative_step,
        dtheta,
        resample2,
    )
    centered_sub_sino, sub_shift = sr.auto_center_sinogram(sub_sinogram)
    sub_sinogram = centered_sub_sino
    recon_sub = sr.reconstruct_focal_spot(sub_sinogram, filter_name, symmetrize)

    # ... (angle finding logic) ...
    angle_step = 360.0 / n_angles
    angles_deg = np.arange(n_angles) * angle_step
    h_idx = np.argmin(np.abs(angles_deg - 0))

    # --- 
    # CORRECTED PLOTTING SECTION
    # ---
    print(f"Generating interpolation plots for angle index {h_idx}...")

    # Get the raw data (blue dots) for the plot
    half_wedge = np.deg2rad(dtheta) / 2
    ys, xs = np.indices(img.shape)
    xs = xs.astype(np.float32) - cx
    ys = ys.astype(np.float32) - cy
    phis = np.arctan2(ys, xs)
    rs = np.hypot(xs, ys) - radius
    min_r = -profile_half_length
    max_r = profile_half_length
    
    internal_angles_rad = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    theta_rad = internal_angles_rad[h_idx]

    dphi = (phis - theta_rad + np.pi) % (2 * np.pi) - np.pi
    mask = np.abs(dphi) <= half_wedge
    r_vals = rs[mask]
    intensities = img[mask]
    radial_mask = (r_vals >= min_r) & (r_vals <= max_r)
    r_vals = r_vals[radial_mask]
    intensities = intensities[radial_mask]

    # Get the radial grid (x-axis)
    r_grid = np.arange(min_r, max_r + 1 / resample2, 1 / resample2)
    
    # ---
    # THE FIX: Get the *actual* calculated profile from your function
    # ---
    calculated_profile = sub_profiles[:, h_idx]

    plt.figure(figsize=(12, 7))
    plt.title(f"Oversampled ESF at Angle {angles_deg[h_idx]:.2f}° (Index {h_idx})")
    
    # Plot the raw pixel data
    plt.scatter(r_vals, intensities, s=10, alpha=0.5, label="Raw Pixel Data (in wedge)")
    
    # Plot the final *calculated* line
    plt.plot(r_grid, calculated_profile, color='green', marker='.', markersize=3, linestyle='-', label="Calculated ESF (Bin+Fill+Smooth)")
    
    # Find and plot "empty" bins (extrapolated points)
    if r_vals.size > 0:
        empty_mask = (r_grid < r_vals.min()) | (r_grid > r_vals.max())
        plt.scatter(r_grid[empty_mask], calculated_profile[empty_mask], 
                    color='red', s=15, label="Empty Bins (Extrapolated)", zorder=5)
    else:
        plt.plot(r_grid, np.zeros_like(r_grid), 'r--', label="All Bins Empty")

    plt.xlabel("Radial Position (pixels)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(r".\oversampled_esf_interpolation.png")
    plt.show()
    # --- END NEW PLOTTING SECTION ---

    print("...continuing with MTF calculations.")
    
    # v_idx = np.argmin(np.abs(angles - 90))  # Closest to 90°
    freq_h, mtf_h, mtf10_h = compute_1d_mtf(
        reconstruction, axis=0, pixel_size=pixel_size
    )
    freq_sino_h, mtf_sino_h, mtf10_sino_h = compute_mtf_from_sinogram(
        sinogram, pixel_size=pixel_size, angle=h_idx
    )

    freq_h_ov, mtf_h_ov, mtf10_h_ov = compute_1d_mtf(
        recon_sub, axis=0, pixel_size=pixel_size / resample2
    )
    freq_sino_h_ov, mtf_sino_h_ov, mtf10_sino_h_ov = compute_mtf_from_sinogram(
        sub_sinogram, pixel_size=pixel_size / resample2, angle=h_idx
    )

    out_path_h = r".\mtf_1d_horizontal.png"
    plot_1d_mtf(
        freq_h, mtf_h, pixel_size, out_path_h, mtf10_freq=mtf10_h, show_plots=True
    )
    
    out_path_sino_h = r".\mtf_1d_horizontal_sino.png"
    plot_1d_mtf(
        freq_sino_h,
        mtf_sino_h,
        pixel_size,
        out_path_sino_h,
        mtf10_freq=mtf10_sino_h,
        show_plots=True,
    )
    
    out_path_h_ov = r".\mtf_1d_horizontal_oversampled.png"
    # Note: For the plot, we pass the *original* pixel size to show the correct Nyquist
    plot_1d_mtf(
        freq_h_ov,
        mtf_h_ov,
        pixel_size, # Show plot relative to original detector Nyquist
        out_path_h_ov,
        mtf10_freq=mtf10_h_ov,
        show_plots=True,
    )
    
    out_path_sino_h_ov = r".\mtf_1d_horizontal_sino_oversampled.png"
    # Note: For the plot, we pass the *original* pixel size to show the correct Nyquist
    plot_1d_mtf(
        freq_sino_h_ov,
        mtf_sino_h_ov,
        pixel_size, # Show plot relative to original detector Nyquist
        out_path_sino_h_ov,
        mtf10_freq=mtf10_sino_h_ov,
        show_plots=True,
    )
    plt.show() # Show the last plot