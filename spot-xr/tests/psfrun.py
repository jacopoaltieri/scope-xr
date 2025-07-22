import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spotxr import utils, plotters
import spotxr.arg_parser_fs as afs
import spotxr.arg_parser_psf as apsf
import spotxr.circle_detection as circ
import spotxr.image_opening as io
import spotxr.mtf_calc as mtfc
import spotxr.sinogram_recon as sr
import spotxr.widths_calculator as wc


def run_pipeline_psf():
    args = apsf.get_merged_config()
    apsf.validate_args(args)

    print("Running detector PSF reconstruction pipeline.")
    print("Arguments in use:")
    for k, v in args.items():
        if k != "hough_params":
            print(f"  {k:18}: {v}")

    # ----------------------------------------------------------------------------------#
    img_path = args["img_path"]
    pixel_size = args.get("pixel_size")  # in mm
    circle_diameter = args.get("circle_diameter")  # in mm
    no_hough = args.get("no_hough")
    n_angles = args.get("n_angles")
    profile_half_length = args.get("profile_half_length")
    derivative_step = args.get("derivative_step")
    filter_name = args.get("filter_name")
    symmetrize = args.get("symmetrize")
    shift_sino = args.get("shift_sino")
    avg_neighbors = args.get("avg_neighbors")
    avg_number = args.get("avg_number")
    oversample = args.get("oversample")
    oversample_strategy = args.get("oversample_strategy")
    dtheta = args.get("dtheta")
    resample1 = args.get("resample1")
    resample2 = args.get("resample2")
    gaussian_sigma = args.get("gaussian_sigma")
    show_plots = args.get("show_plots")

    # ----------------------------------------------------------------------------------#
    # Create output directory
    basename = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(args.get("out_dir", "."), basename)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving outputs to {out_dir}")

    # Load image
    try:
        img = io.load_image(img_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to load image at `{img_path}`: {e}")

    # Circle detection
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
            debug=args["hough_params"].get("debug", False),
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

    # Extract profiles and sinogram
    profiles, sinogram = sr.compute_profiles_and_sinogram(
        cropped, cx, cy, radius, n_angles, profile_half_length, derivative_step
    )

    # Center sinogram if requested
    if shift_sino:
        centered_sino, applied_shift = sr.auto_center_sinogram(sinogram)
        sinogram = centered_sino
        print(f"Applied axis shift: {applied_shift} px")

    reconstruction = sr.reconstruct_focal_spot(sinogram, filter_name, symmetrize)

    # Save and plot function
    saved_files = []

    def save_and_plot(name, arr, plot_func=None, suffix=""):
        fname = f"{name}{suffix}.png" if not name.endswith(".png") else name
        path = os.path.join(out_dir, fname)
        plt.imsave(path, arr, cmap="gray")
        saved_files.append(path)
        if plot_func:
            plot_func(arr, out_dir, show_plots)
        return path

    # Base saving
    save_and_plot("profiles", profiles)
    save_and_plot("sinogram", sinogram)
    save_and_plot("reconstruction", reconstruction)

    plotters.plot_profiles_and_reconstruction(
        profiles,
        sinogram,
        reconstruction,
        out_dir,
        show_plots,
        reconstruction_type="psf",
    )

    # Find horizontal and vertical profiles
    angle_step = 360.0 / n_angles
    angles = np.arange(n_angles) * angle_step

    h_idx = np.argmin(np.abs(angles - 0))  # Closest to 0°
    v_idx = np.argmin(np.abs(angles - 90))  # Closest to 90°

    _, _, sigmas, pops = wc.find_extreme_profiles_gaussian(sinogram)
    # Get profiles for h and v angles
    if avg_neighbors:
        prof_h_sino = wc.average_neighbors(sinogram, h_idx, avg_number)
        prof_v_sino = wc.average_neighbors(sinogram, v_idx, avg_number)
    else:
        prof_h_sino = sinogram[:, h_idx]
        prof_v_sino = sinogram[:, v_idx]

    popt_h = pops[h_idx]
    popt_v = pops[v_idx]
    fw_h = wc.fwhm_from_sigma(sigmas[h_idx])
    fw_v = wc.fwhm_from_sigma(sigmas[v_idx])
    print(f"Horizontal:   FWHM={fw_h:.2f}px")
    print(f"Vertical: FWHM={fw_v:.2f}px")

    radial = np.arange(sinogram.shape[0]) - (sinogram.shape[0] // 2)

    # Plot profiles with Gaussian fits
    plotters.plot_profile_with_gaussian(
        radial=radial,
        sinogram_profile=prof_h_sino,
        popt=popt_h,
        out_path=os.path.join(out_dir, "sinogram_profile_horizontal.png"),
        show_plots=show_plots,
    )
    plotters.plot_profile_with_gaussian(
        radial=radial,
        sinogram_profile=prof_v_sino,
        popt=popt_v,
        out_path=os.path.join(out_dir, "sinogram_profile_vertical.png"),
        show_plots=show_plots,
    )

    # Plot sinogram and reconstruction with lines
    plotters.plot_sinogram_with_traced_profiles(
        sinogram,
        h_idx,
        v_idx,
        os.path.join(out_dir, "sinogram_traced_profiles.png"),
        reconstruction_type="psf",
        show_plots=show_plots,
    )
    plotters.plot_recon_with_lines(
        reconstruction,
        h_idx,
        v_idx,
        os.path.join(out_dir, "psf_traced_profiles.png"),
        show_plots=show_plots,
        reconstruction_type="psf",
    )

    # Compute MTF in horizontal and vertical directions
    freq_h, mtf_h, mtf10_h = mtfc.compute_1d_mtf(
        reconstruction, axis=0, pixel_size=pixel_size
    )
    freq_v, mtf_v, mtf10_v = mtfc.compute_1d_mtf(
        reconstruction, axis=1, pixel_size=pixel_size
    )

    print(f"MTF10 horizontal: {mtf10_h:.3f} cycles/mm")
    print(f"MTF10 vertical:   {mtf10_v:.3f} cycles/mm")

    plotters.plot_1d_mtf(
        freq_h,
        mtf_h,
        pixel_size=pixel_size,
        out_path=os.path.join(out_dir, "mtf_horizontal.png"),
        mtf10_freq=mtf10_h,
        show_plots=show_plots,
    )
    plotters.plot_1d_mtf(
        freq_v,
        mtf_v,
        pixel_size=pixel_size,
        out_path=os.path.join(out_dir, "mtf_vertical.png"),
        mtf10_freq=mtf10_v,
        show_plots=show_plots,
    )

    # Prepare summary
    summary = [
        f"Output saved to: {out_dir}",
        f"Arguments: {args}",
        f"COM circle: center=({cx},{cy}), radius={radius}px",
        f"PSF size px:     horizontal={fw_h:.3f}, vertical={fw_v:.3f}",
        f"MTF10 horizontal: {mtf10_h:.3f} cycles/mm",
        f"MTF10 vertical:   {mtf10_v:.3f} cycles/mm",
    ]

    # ----------------------------------------------------------------------------------#
    # Oversampling section
    if oversample:
        if oversample_strategy == 1:
            sub_profiles, sub_sinogram = (
                sr.compute_subpixel_profiles_and_sinogram_traditional(
                    cropped,
                    cx,
                    cy,
                    radius,
                    n_angles,
                    profile_half_length,
                    derivative_step,
                    dtheta,
                    resample2,
                )
            )
            print("Using oversampling strategy 1 (traditional).")

        elif oversample_strategy == 2:
            sub_profiles, sub_sinogram = (
                sr.compute_subpixel_profiles_and_sinogram_3step(
                    cropped,
                    cx,
                    cy,
                    radius,
                    n_angles,
                    profile_half_length,
                    derivative_step,
                    dtheta,
                    gaussian_sigma,
                    resample1,
                    resample2,
                )
            )
            print("Using oversampling strategy 2 (3-step).")
        else:
            raise ValueError(f"Invalid oversample strategy: {oversample_strategy}")

        if shift_sino:
            centered_sub_sino, sub_shift = sr.auto_center_sinogram(sub_sinogram)
            sub_sinogram = centered_sub_sino
            print(f"Applied axis shift (oversampled): {sub_shift} px")

        save_and_plot("profiles_oversampled", sub_profiles)
        save_and_plot("sinogram_oversampled", sub_sinogram)
        recon_sub = sr.reconstruct_focal_spot(sub_sinogram, filter_name, symmetrize)

        # Save oversampled images
        save_and_plot("reconstruction_oversampled", recon_sub)
        plotters.plot_profiles_and_reconstruction(
            sub_profiles,
            sub_sinogram,
            recon_sub,
            out_dir,
            show_plots,
            reconstruction_type="psf",
            suffix="_oversampled",
        )

        # Find extreme profiles oversampled
        _, _, sigmas_ov, pops_ov = wc.find_extreme_profiles_gaussian(sub_sinogram)
        # Get profiles for h and v angles
        if avg_neighbors:
            prof_h_sino_ov = wc.average_neighbors(sub_sinogram, h_idx, avg_number)
            prof_v_sino_ov = wc.average_neighbors(sub_sinogram, v_idx, avg_number)
        else:
            prof_h_sino_ov = sub_sinogram[:, h_idx]
            prof_v_sino_ov = sub_sinogram[:, v_idx]

        popt_h_ov = pops_ov[h_idx]
        popt_v_ov = pops_ov[v_idx]
        fw_h_ov = wc.fwhm_from_sigma(sigmas_ov[h_idx]) * resample2
        fw_v_ov = wc.fwhm_from_sigma(sigmas_ov[v_idx]) * resample2
        print(f"Horizontal:   FWHM={fw_h_ov:.2f}px")
        print(f"Vertical: FWHM={fw_v_ov:.2f}px")

        radial_ov = (
            np.arange(sub_sinogram.shape[0]) - (sub_sinogram.shape[0] // 2)
        ) * resample2

        plotters.plot_profile_with_gaussian(
            radial=radial_ov,
            sinogram_profile=prof_h_sino_ov,
            popt=popt_h_ov,
            out_path=os.path.join(
                out_dir, "oversampled_sinogram_profile_horizontal.png"
            ),
            show_plots=show_plots,
        )
        plotters.plot_profile_with_gaussian(
            radial=radial_ov,
            sinogram_profile=prof_v_sino_ov,
            popt=popt_v_ov,
            out_path=os.path.join(out_dir, "oversampled_sinogram_profile_vertical.png"),
            show_plots=show_plots,
        )

        plotters.plot_sinogram_with_traced_profiles(
            sub_sinogram,
            h_idx,
            v_idx,
            os.path.join(out_dir, "oversampled_sinogram_traced_profiles.png"),
            reconstruction_type="psf",
            show_plots=show_plots,
        )
        plotters.plot_recon_with_lines(
            recon_sub,
            h_idx,
            v_idx,
            os.path.join(out_dir, "psf_traced_profiles_oversampled.png"),
            show_plots=show_plots,
            reconstruction_type="psf",
        )

        # Compute MTF in horizontal and vertical directions
        freq_h_ov, mtf_h_ov, mtf10_h_ov = mtfc.compute_1d_mtf(
            recon_sub, axis=0, pixel_size=pixel_size * resample2
        )
        freq_v_ov, mtf_v_ov, mtf10_v_ov = mtfc.compute_1d_mtf(
            recon_sub, axis=1, pixel_size=pixel_size * resample2
        )

        print(f"MTF10 horizontal oversampled: {mtf10_h:.3f} cycles/mm")
        print(f"MTF10 vertical oversampled:   {mtf10_v:.3f} cycles/mm")

        plotters.plot_1d_mtf(
            freq_h_ov,
            mtf_h_ov,
            pixel_size=pixel_size,
            out_path=os.path.join(out_dir, "mtf_horizontal_oversampled.png"),
            mtf10_freq=mtf10_h_ov,
            show_plots=show_plots,
        )
        plotters.plot_1d_mtf(
            freq_v_ov,
            mtf_v_ov,
            pixel_size=pixel_size,
            out_path=os.path.join(out_dir, "mtf_vertical_oversampled.png"),
            mtf10_freq=mtf10_v_ov,
            show_plots=show_plots,
        )

        # Append oversampled summary
        summary += [
            "Oversampled results:",
            f"PSF size px (oversampled):     horizontal={fw_h_ov:.3f}, vertical={fw_v_ov:.3f}",
            f"MTF10 horizontal oversampled: {mtf10_h_ov:.3f} cycles/mm",
            f"MTF10 vertical oversampled:   {mtf10_v_ov:.3f} cycles/mm",
        ]

    # Save summary to txt
    results_path = os.path.join(out_dir, "psf_results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(summary))
    print(f"Results written to {results_path}")

 
if __name__ == "__main__":
    run_pipeline_psf()