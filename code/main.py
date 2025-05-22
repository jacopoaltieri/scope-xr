import os
import numpy as np
import matplotlib.pyplot as plt

import image_opening as io
import circle_detection as circ
import sinogram_recon as sr
import plotters
import widths_calculator as wc
from arg_parser import get_args


def eval_minimum_magnification(a: int, n: int, p: int) -> int:
    """Evaluate the minimum magnification required to obtain a focal spot image involving a reasonable number n of pixels."""
    m = (a + n * p) / a
    return m


def eval_minimum_radius(n: int, p: int, m: int) -> int:
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
            cmap="gray"
        )
        print(f"Saved cropped image to '{output_path}\cropped.png'")
    return cropped




if __name__ == "__main__":
    # ----------------------------------------------------------------------------------#
    # Args default to this if not passed via command line
    default_args = {
        "img_path": r"C:\Users\jacop\Desktop\PhD\Focal Spot\ATS 6 mag\P146\00013_8ca7056d-d871-493e-ad31-99ce37ec2c6d_20250506_120021.2359.raw",
        "out_dir": r".\output",
        "show_plots": False,
        "use_hough": True,
        "pixel_size": 0.154,
        "circle_diameter": 10,
        "min_n": 6,
        "n_angles": 360,
        "profile_half_length": 64,
        "derivative_step": 1,
        "filter_name": "hann",
        "symmetrize": False,
        "shift_sino": True,
    }

    # ----------------------------------------------------------------------------------#
    args = get_args(**default_args)
    print("Arguments in use:")
    for k, v in args.items():
        print(f"  {k:18}: {v}")

    # ----------------------------------------------------------------------------------#
    img_path = args["img_path"]
    out_dir = args["out_dir"]
    show_plots = args["show_plots"]
    use_hough = args["use_hough"]
    pixel_size = args["pixel_size"]
    circle_diameter = args["circle_diameter"]
    min_n = args["min_n"]
    n_angles = args["n_angles"]
    profile_half_length = args["profile_half_length"]
    derivative_step = args["derivative_step"]
    filter_name = args["filter_name"]
    symmetrize = args["symmetrize"]
    shift_sino = args["shift_sino"]

    # ----------------------------------------------------------------------------------#
    os.makedirs(out_dir, exist_ok=True)
    img = io.load_image(img_path)

    if use_hough:
        # Detect circle using Hough Transform
        hough_circle = circ.detect_circle_hough(
            img,
            dp=1.3,
            min_dist=100,
            param1=70,
            param2=20,
            min_radius=10,
            max_radius=0,
            debug=False,  # if True show the plot, but freezes the script
        )

        if hough_circle:
            x, y, r = hough_circle
            print(
                f"Estimated circle via Hough transform: Center=({x}, {y}), Radius={r} px"
            )

            # Crop image around the detected circle
            cropped = crop_square_roi(
                img, center=(x, y), radius=r, width_factor=1.5, output_path=out_dir
            )
        else:
            raise ValueError(
                "Hough transform did not detect any circle. PLease upload an already cropped image."
            )
    else:
        print(
            "Warning: Hough transform not used. Using provided image as already cropped."
        )
        cropped = img

    cx, cy, radius = circ.estimate_circle(cropped)
    print(
        f"Estimated circle via Center Of Mass: Center=({cx}, {cy}), Radius={radius} px"
    )
    plotters.plot_circle_on_crop(cropped, cx, cy, radius, out_dir, show_plots)

    m = radius / (circle_diameter / 2)  # magnification
    min_r = eval_minimum_radius(min_n, pixel_size, m)
    if min_r > radius:
        print(
            f"Warning: The estimated radius {radius} mm is smaller than the minimum required radius {min_r:.2f} mm."
        )

    # Extract profiles and sinogram
    profiles, sinogram = sr.compute_profiles_and_sinogram(cropped, cx, cy, radius)
    if shift_sino:
        best_centered_sino, applied_shift = sr.auto_center_sinogram(sinogram)
        sinogram = best_centered_sino[applied_shift:-applied_shift, :]
        print(f"Applied axis shift: {applied_shift} px")

    reconstruction = sr.reconstruct_focal_spot(sinogram, filter_name, symmetrize)

    profiles_path = os.path.join(out_dir, "profiles.png")
    plt.imsave(profiles_path, profiles, cmap="gray", origin="lower")
    print(f"Saved profiles to {profiles_path}")

    # 2) Save the sinogram
    sinogram_path = os.path.join(out_dir, "sinogram.png")
    plt.imsave(sinogram_path, sinogram, cmap="gray", origin="lower")
    print(f"Saved sinogram to {sinogram_path}")

    recon_path = os.path.join(out_dir, "reconstruction.png")
    plt.imsave(recon_path, reconstruction, cmap="gray")
    print(f"Saved reconstruction to {recon_path}")

    plotters.plot_profiles_and_reconstruction(
        profiles, sinogram, reconstruction, out_dir, show_plots
    )

    # Shift the central axis and save as a sequence. This is useful to see if the centering is correct.
    axis_shifts = list(range(-10, 11))
    shift_tiff_path = os.path.join(out_dir, "recon_axis_shifts.tiff")
    sr.reconstruct_with_axis_shifts(
        sinogram, shift_tiff_path, filter_name, shifts=axis_shifts
    )

    wide_idx, narrow_idx, slopes = wc.find_extreme_profiles_erf(profiles)
    print(f"Widest edge at angle idx {wide_idx} (slope={slopes[wide_idx]:.3f})")
    print(f"Narrowest edge at angle idx {narrow_idx} (slope={slopes[narrow_idx]:.3f})")

    prof_wide_sino = sinogram[:, wide_idx]
    prof_narrow_sino = sinogram[:, narrow_idx]

    fw, lw, rw = wc.fwhm(prof_wide_sino)
    fn, ln, rn = wc.fwhm(prof_narrow_sino)
    print(f"Widest:   FWHM={fw}px (from {lw} to {rw})")
    print(f"Narrowest: FWHM={fn}px (from {ln} to {rn})")

    n_rays = sinogram.shape[0]
    radial = np.arange(n_rays) - n_rays // 2

    fwhm_path = (os.path.join(out_dir, "fwhm_sinogram_profiles.png"))
    plotters.plot_profiles_with_fwhm(
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
        fwhm_path,
        show_plots,
    )

    sino_with_lines_path = (os.path.join(out_dir, "sinogram_traced_profiles.png"))
    plotters.plot_sinogram_with_traced_profiles(
        sinogram, wide_idx, narrow_idx, sino_with_lines_path, show_plots
    )

    angle_step = 360.0 / n_angles
    angle_wide_deg   = wide_idx   * angle_step
    angle_narrow_deg = narrow_idx * angle_step
    spot_with_lines_path = (os.path.join(out_dir, "focal_spot_traced_profiles.png"))
    plotters.plot_focal_spot_with_lines(
        reconstruction,
        angle_wide_deg,
        angle_narrow_deg,
        out_path=spot_with_lines_path,
        show_plots=show_plots,
    )
