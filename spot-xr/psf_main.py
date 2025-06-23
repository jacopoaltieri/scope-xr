import os
import numpy as np
import matplotlib.pyplot as plt

from psf.arg_parser import get_merged_config, validate_args
from spotx import image_opening as io
from spotx import circle_detection as circ
from spotx import utils, plotters
from spotx import sinogram_recon as sr
from spotx import widths_calculator as wc

if __name__ == "__main__":
    args = get_merged_config()
    validate_args(args)

    print("Arguments in use:")
    for k, v in args.items():
        if k != "hough_params":
            print(f"  {k:18}: {v}")

    # ----------------------------------------------------------------------------------#
    img_path = args["img_path"]
    pixel_size = args["pixel_size"]  # in mm
    circle_diameter = args["circle_diameter"]  # in mm
    no_hough = args["no_hough"]
    n_angles = args["n_angles"]
    profile_half_length = args["profile_half_length"]
    derivative_step = args["derivative_step"]
    filter_name = args["filter_name"]
    symmetrize = args["symmetrize"]
    shift_sino = args["shift_sino"]
    show_plots = args["show_plots"]
    dtheta = args.get("dtheta", 5)
    gaussian_sigma = args.get("gaussian_sigma", 0.2)

    # ----------------------------------------------------------------------------------#
    # create output directory
    basename = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(args["out_dir"], basename)
    os.makedirs(out_dir, exist_ok=True)
    print(f"saving outputs to {out_dir}")

    # load the image
    try:
        img = io.load_image(img_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to load image at `{img_path}`: {e}")

    # circle detection or cropping
    if no_hough:
        print(
            "Caution! Hough transform not used. Using provided image as already cropped."
        )
        cropped = img
    else:
        hough_circle = circ.detect_circle_hough(
            img,
            dp=args["hough_params"]["dp"],
            min_dist=args["hough_params"]["min_dist"],
            param1=args["hough_params"]["param1"],
            param2=args["hough_params"]["param2"],
            min_radius=args["hough_params"]["min_radius"],
            max_radius=args["hough_params"]["max_radius"],
            output_path=out_dir,
            debug=args["hough_params"]["debug"],
        )
        if not hough_circle:
            raise ValueError(
                "Hough transform did not detect any circle. Provide a cropped image."
            )
        x, y, r = hough_circle
        print(f"Detected circle via Hough transform: Center=({x}, {y}), Radius={r} px")
        cropped = utils.crop_square_roi(
            img, center=(x, y), radius=r, width_factor=1.5, output_path=out_dir
        )

    cx, cy, radius = circ.estimate_circle(cropped)
    
    if not circ.is_circle_centered(cropped, cx, cy):
        print("Warning: The estimated circle center is not at the image center.")
        exit(1)
    print(
        f"Estimated circle via Center Of Mass: Center=({cx}, {cy}), Radius={radius} px"
    )
    plotters.plot_circle_on_crop(cropped, cx, cy, radius, out_dir, show_plots)

    # ------------------------------------------------------------------------#
    # 1) Original pixel-based profiles and sinogram
    profiles, sinogram = sr.compute_profiles_and_sinogram(
        cropped, cx, cy, radius,
        n_angles, profile_half_length, derivative_step
    )
    # Save original
    plt.imsave(os.path.join(out_dir, "profiles_px.png"), profiles, cmap="gray")
    plt.imsave(os.path.join(out_dir, "sinogram_px.png"), sinogram, cmap="gray")

    # 2) Subpixel ESF-based profiles and sinogram
    sub_profiles, sub_sinogram = sr.compute_subpixel_profiles_and_sinogram(
        cropped, cx, cy, radius,
        n_angles,
        profile_half_length,
        derivative_step,
        dtheta=dtheta,
        gaussian_sigma=gaussian_sigma,
    )
    # Save subsampled
    plt.imsave(os.path.join(out_dir, "profiles_subpx.png"), sub_profiles, cmap="gray")
    plt.imsave(os.path.join(out_dir, "sinogram_subpx.png"), sub_sinogram, cmap="gray")

    # Apply optional centering
    def process_sino(sino):
        if shift_sino:
            centered, shift = sr.auto_center_sinogram(sino)
            if shift != 0:
                sino = centered[shift:-shift, :]
            else:
                sino = centered
        return sino


    sinogram = process_sino(sinogram)
    sub_sinogram = process_sino(sub_sinogram)

    
    # Reconstructions
    recon_px = sr.reconstruct_focal_spot(sinogram, filter_name, symmetrize)
    recon_sub = sr.reconstruct_focal_spot(sub_sinogram, filter_name, symmetrize)
    plt.imsave(os.path.join(out_dir, "reconstruction_px.png"), recon_px, cmap="gray")
    plt.imsave(os.path.join(out_dir, "reconstruction_subpx.png"), recon_sub, cmap="gray")

    # Plot combined
    plotters.plot_profiles_and_reconstruction(
        profiles, sinogram, recon_px, out_dir, show_plots
    )


    # ------------------------------------------------------------------------#
    # Width analysis (classic and Gaussian) for both methods
# Example: assume resample2 is defined earlier in your code; e.g.
# resample2 = 0.02  # radial spacing of subpixel sinogram in detector-pixel units
    summary_lines = [f"Output dir: {out_dir}"]

    for label, sino in [("px", sinogram), ("subpx", sub_sinogram)]:
        n_radial= sino.shape[0]
        center = n_radial // 2

        # Determine spacing per sample in detector-pixel units
        if label == "px":
            spacing = 1.0
        else:
            spacing = 0.02

        # Find extreme profiles and Gaussian fit params
        w_idx, n_idx, sigmas, pops = wc.find_extreme_profiles_gaussian(sino)
        # Build radial axis arrays
        # For “px”: each index step is 1 px → radial positions in px
        # For “subpx”: each index step is resample2 px → radial positions in px
        radial_indices = np.arange(n_radial) - center
        if label == "px":
            radial = radial_indices.astype(float)
        else:
            radial = radial_indices.astype(float) * spacing

        # Extract profiles
        prof_w = sino[:, w_idx]
        prof_n = sino[:, n_idx]

        # Get Gaussian-fit parameters for these profiles
        popt_w = pops[w_idx]  # expected shape (4,)
        popt_n = pops[n_idx]
        print(f"Fitting parameters for {label} profiles: wide={popt_w[2]}, narrow={popt_n[2]}")
        # Plot and save figures for wide profile
        out_path_w = os.path.join(out_dir, f"profile_{label}_wide.png")
        plotters.plot_profile_with_gaussian(
            radial=radial,
            sinogram_profile=prof_w,
            popt=popt_w,
            out_path=out_path_w,
            show_plots=show_plots,
        )
        # Plot and save figures for narrow profile
        out_path_n = os.path.join(out_dir, f"profile_{label}_narrow.png")
        plotters.plot_profile_with_gaussian(
            radial=radial,
            sinogram_profile=prof_n,
            popt=popt_n,
            out_path=out_path_n,
            show_plots=show_plots,
        )

        # Compute FWHM in sample units
        fw_samples, _, _ = wc.fwhm(prof_w)
        fn_samples, _, _ = wc.fwhm(prof_n)
        fw_g_samples = wc.fwhm_from_sigma(sigmas[w_idx])
        fn_g_samples = wc.fwhm_from_sigma(sigmas[n_idx])

        # Convert to detector-pixel units
        fw_phys = fw_samples * spacing
        fn_phys = fn_samples * spacing
        fw_g_phys = fw_g_samples * spacing
        fn_g_phys = fn_g_samples * spacing

        sigma_w_phys = sigmas[w_idx] * spacing
        sigma_n_phys = sigmas[n_idx] * spacing

        # Print results
        print(f"[{label}] Wide idx {w_idx}:")
        print(f"    - FWHM (raw profile)     = {fw_samples:.2f} samples  → {fw_phys:.2f} detector px")
        print(f"    - FWHM (Gaussian fit)    = {fw_g_samples:.2f} samples  → {fw_g_phys:.2f} detector px")
        print(f"    - Sigma from fit         = {sigmas[w_idx]:.3f} samples  → {sigma_w_phys:.3f} detector px")
        print(f"[{label}] Narrow idx {n_idx}:")
        print(f"    - FWHM (raw profile)     = {fn_samples:.2f} samples  → {fn_phys:.2f} detector px")
        print(f"    - FWHM (Gaussian fit)    = {fn_g_samples:.2f} samples  → {fn_g_phys:.2f} detector px")
        print(f"    - Sigma from fit         = {sigmas[n_idx]:.3f} samples  → {sigma_n_phys:.3f} detector px")
        print("-" * 60)

        # Append to summary lines (you may choose which values to record)
        if label == "px":
            summary_lines.append(f"Classic FWHM px (wide) = {fw_phys:.2f}, (narrow) = {fn_phys:.2f}")
        else:
            summary_lines.append(f"Subpx FWHM px (wide) = {fw_g_phys:.2f}, (narrow) = {fn_g_phys:.2f}")

    # Write summary to results.txt
    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Results written to {out_dir}/results.txt")