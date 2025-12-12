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
import os


from . import utils, plotters
from . import arg_parser_psf as apsf
from . import circle_detection as circ
from . import image_opening as io
from . import mtf_calc as mtfc
from . import sinogram_recon as sr
from . import widths_calculator as wc

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
    axis_shifts = args["axis_shifts"]
    filter_name = args.get("filter_name")
    symmetrize = args.get("symmetrize")
    manual_shift = args["manual_shift"]
    auto_shift = args["auto_shift"]
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
            img, center=(x, y), radius=r, width_factor=1.2, output_path=out_dir
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
    if manual_shift is not None:
        print(f"Applying manual shift: {manual_shift} px")
        centered_sino, applied_shift = sr.manual_center_sinogram(sinogram, manual_shift)
        sinogram = centered_sino
    elif auto_shift:
        print("Running automatic sinogram centering...")
        centered_sino, applied_shift = sr.auto_center_sinogram(sinogram)
        sinogram = centered_sino
        print(f"Applied automatic axis shift: {applied_shift} px")

    else:
        # 3. No shift is applied (no_shift: True or all are False)
        applied_shift = 0
        print("Sinogram shifting is disabled.")

    reconstruction = sr.reconstruct_focal_spot(sinogram, filter_name, symmetrize)

    # Shift the central axis and save as a sequence. This is useful to see if the centering is correct.
    shift_list = list(range(-axis_shifts, axis_shifts))
    shift_tiff_path = os.path.join(out_dir, "recon_axis_shifts.tiff")
    sr.reconstruct_with_axis_shifts(
        sinogram, shift_tiff_path, filter_name, shifts=shift_list
    )

    utils.save_and_plot("profiles", profiles,  out_dir),
    utils.save_and_plot("sinogram", sinogram, out_dir),
    utils.save_and_plot("reconstruction", reconstruction,  out_dir),

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

    # NOTE: The reconstruction introduces a filtering effect, so we compute MTF directly from sinogram
    # freq_h, mtf_h, mtf10_h = mtfc.compute_1d_mtf(
    #     reconstruction, axis=0, pixel_size=pixel_size
    # )
    # freq_v, mtf_v, mtf10_v = mtfc.compute_1d_mtf(
    #     reconstruction, axis=1, pixel_size=pixel_size
    # )
    freq_h, mtf_h, mtf10_h = mtfc.compute_1d_mtf_from_sino(sinogram, pixel_size, h_idx)

    mtf1_h = mtfc.get_mtf_at_freq(1.0, freq_h, mtf_h)
    mtf2_h = mtfc.get_mtf_at_freq(2.0, freq_h, mtf_h)
    mtf3_h = mtfc.get_mtf_at_freq(3.0, freq_h, mtf_h)

    freq_v, mtf_v, mtf10_v = mtfc.compute_1d_mtf_from_sino(sinogram, pixel_size, v_idx)

    mtf1_v = mtfc.get_mtf_at_freq(1.0, freq_v, mtf_v)
    mtf2_v = mtfc.get_mtf_at_freq(2.0, freq_v, mtf_v)
    mtf3_v = mtfc.get_mtf_at_freq(3.0, freq_v, mtf_v)

    print(f"Horizontal MTF10: {mtf10_h:.3f} cycles/mm")
    print(f"Horizontal MTF(1.0 c/mm) = {mtf1_h:.3f}")
    print(f"Horizontal MTF(2.0 c/mm) = {mtf2_h:.3f}")
    print(f"Horizontal MTF(3.0 c/mm) = {mtf3_h:.3f}")

    print(f"Vertical MTF10:   {mtf10_v:.3f} cycles/mm")
    print(f"Vertical MTF(1.0 c/mm) = {mtf1_v:.3f}")
    print(f"Vertical MTF(2.0 c/mm) = {mtf2_v:.3f}")
    print(f"Vertical MTF(3.0 c/mm) = {mtf3_v:.3f}")

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
    label_width = 24

    summary = [
        "========================================",
        "  SCOPE-XR PSF Analysis Results",
        "========================================",
        "",
        f"Full arguments: {args}",  # Good for traceability
        "--- General Info ---",
        f"{'Input Image:': <{label_width}} {os.path.basename(img_path)}",
        f"{'Output Directory:': <{label_width}} {out_dir}",
        "",
        "--- Setup Parameters ---",
        f"{'COM Circle Center:': <{label_width}} ({cx:.2f}, {cy:.2f}) px",
        f"{'COM Circle Radius:': <{label_width}} {radius:.2f} px",
        "",
        "--- PSF Size (FWHM from Sinogram) ---",
        f"{'FWHM Horizontal:': <{label_width}} {fw_h:.3f} px",
        f"{'FWHM Vertical:': <{label_width}} {fw_v:.3f} px",
        "",
        "--- MTF Horizontal (from Sinogram) ---",
        f"{'MTF10:': <{label_width}} {mtf10_h:.3f} cycles/mm",
        f"{'MTF @ 1.0 cy/mm:': <{label_width}} {mtf1_h:.3f}",
        f"{'MTF @ 2.0 cy/mm:': <{label_width}} {mtf2_h:.3f}",
        f"{'MTF @ 3.0 cy/mm:': <{label_width}} {mtf3_h:.3f}",
        "",
        "--- MTF Vertical (from Sinogram) ---",
        f"{'MTF10:': <{label_width}} {mtf10_v:.3f} cycles/mm",
        f"{'MTF @ 1.0 cy/mm:': <{label_width}} {mtf1_v:.3f}",
        f"{'MTF @ 2.0 cy/mm:': <{label_width}} {mtf2_v:.3f}",
        f"{'MTF @ 3.0 cy/mm:': <{label_width}} {mtf3_v:.3f}",
        "",
    ]

    # ----------------------------------------------------------------------------------#
    # Oversampling section
    if oversample:
        max_os_angle = utils.suggest_os_angle(pixel_size, resample2, radius)
        print(
            f"Suggested maximum oversampling angle to avoid cross-talk: {max_os_angle:.2f}°"
        )
        if dtheta > max_os_angle:
            print(
                f"Caution!: The provided oversampling angle {dtheta}° is larger than the suggested maximum {max_os_angle:.2f}°. This may cause cross-talk between neighboring profiles."
            )

        if oversample_strategy == 1:
            print("Using oversampling strategy 1 (traditional).")
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

        elif oversample_strategy == 2:
            print("Using oversampling strategy 2 (3-step).")
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
        else:
            raise ValueError(f"Invalid oversample strategy: {oversample_strategy}")

        applied_shift_ov = 0 # New variable for oversampled shift
        if manual_shift is not None:
            # Scale the manual shift (which is in 'normal' pixels)
            manual_shift_ov = int(manual_shift * resample2)
            print(f"Applying manual shift to oversampled sinogram: {manual_shift_ov} px")
            # Apply shift to sub_sinogram
            centered_sino, applied_shift_ov = sr.manual_center_sinogram(
                sub_sinogram, manual_shift_ov 
            )
            sub_sinogram = centered_sino
        elif auto_shift:
            print("Running automatic sinogram centering (oversampled)...")
            # Apply auto-shift to sub_sinogram
            centered_sino, applied_shift_ov = sr.auto_center_sinogram(sub_sinogram)
            sub_sinogram = centered_sino
            print(f"Applied automatic axis shift: {applied_shift_ov} px (oversampled)")
        else:
            # No shift is applied
            applied_shift_ov = 0
            print("Sinogram shifting is disabled (oversampled).")
            
        
        recon_sub = sr.reconstruct_focal_spot(sub_sinogram, filter_name, symmetrize)
        
        utils.save_and_plot("profiles_oversampled", sub_profiles, out_dir)
        utils.save_and_plot("sinogram_oversampled", sub_sinogram, out_dir)
        utils.save_and_plot("reconstruction_oversampled", recon_sub, out_dir)
        
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
        
        # FWHM value from oversampled (in 'oversampled pixels')
        fw_h_ov_native = wc.fwhm_from_sigma(sigmas_ov[h_idx]) 
        fw_v_ov_native = wc.fwhm_from_sigma(sigmas_ov[v_idx])
        # Convert FWHM to 'normal' pixel-equivalent
        fw_h_ov = fw_h_ov_native / resample2 
        fw_v_ov = fw_v_ov_native / resample2
        
        print(f"Horizontal (Oversampled):  FWHM={fw_h_ov:.2f} px")
        print(f"Vertical (Oversampled): FWHM={fw_v_ov:.2f} px")

        # The radial axis for oversampled plot
        radial_ov = (
            np.arange(sub_sinogram.shape[0]) - (sub_sinogram.shape[0] // 2)
        )

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

        # Compute MTF
        freq_h_ov, mtf_h_ov, mtf10_h_ov = mtfc.compute_1d_mtf_from_sino(
            sub_sinogram, pixel_size / resample2, h_idx
        )

        mtf1_h_ov = mtfc.get_mtf_at_freq(1.0, freq_h_ov, mtf_h_ov)
        mtf2_h_ov = mtfc.get_mtf_at_freq(2.0, freq_h_ov, mtf_h_ov)
        mtf3_h_ov = mtfc.get_mtf_at_freq(3.0, freq_h_ov, mtf_h_ov)

        freq_v_ov, mtf_v_ov, mtf10_v_ov = mtfc.compute_1d_mtf_from_sino(
            sub_sinogram, pixel_size / resample2, v_idx
        )

        mtf1_v_ov = mtfc.get_mtf_at_freq(1.0, freq_v_ov, mtf_v_ov)
        mtf2_v_ov = mtfc.get_mtf_at_freq(2.0, freq_v_ov, mtf_v_ov)
        mtf3_v_ov = mtfc.get_mtf_at_freq(3.0, freq_v_ov, mtf_v_ov)

        print(f"Horizontal oversampled MTF10: {mtf10_h_ov:.3f} cycles/mm")
        print(f"Horizontal oversampled MTF(1.0 c/mm) = {mtf1_h_ov:.3f}")
        print(f"Horizontal oversampled MTF(2.0 c/mm) = {mtf2_h_ov:.3f}")
        print(f"Horizontal oversampled MTF(3.0 c/mm) = {mtf3_h_ov:.3f}")

        print(f"Vertical oversampled MTF10:  {mtf10_v_ov:.3f} cycles/mm")
        print(f"Vertical oversampled MTF(1.0 c/mm) = {mtf1_v_ov:.3f}")
        print(f"Vertical oversampled MTF(2.0 c/mm) = {mtf2_v_ov:.3f}")
        print(f"Vertical oversampled MTF(3.0 c/mm) = {mtf3_v_ov:.3f}")

        plotters.plot_1d_mtf(
            freq_h_ov,
            mtf_h_ov,
            pixel_size=pixel_size, # Plot against original Nyquist
            out_path=os.path.join(out_dir, "mtf_horizontal_oversampled.png"),
            mtf10_freq=mtf10_h_ov,
            show_plots=show_plots,
        )
        plotters.plot_1d_mtf(
            freq_v_ov,
            mtf_v_ov,
            pixel_size=pixel_size, # Plot against original Nyquist
            out_path=os.path.join(out_dir, "mtf_vertical_oversampled.png"),
            mtf10_freq=mtf10_v_ov,
            show_plots=show_plots,
        )

        # Append oversampled summary
        summary += [
            "",
            "--- PSF Size (Oversampled FWHM) ---",
            f"{'FWHM Horizontal:': <{label_width}} {fw_h_ov:.3f} px (native: {fw_h_ov_native:.3f})",
            f"{'FWHM Vertical:': <{label_width}} {fw_v_ov:.3f} px (native: {fw_v_ov_native:.3f})",
            "",
            "--- MTF Horizontal (Oversampled) ---",
            f"{'MTF10:': <{label_width}} {mtf10_h_ov:.3f} cycles/mm",
            f"{'MTF @ 1.0 cy/mm:': <{label_width}} {mtf1_h_ov:.3f}",
            f"{'MTF @ 2.0 cy/mm:': <{label_width}} {mtf2_h_ov:.3f}",
            f"{'MTF @ 3.0 cy/mm:': <{label_width}} {mtf3_h_ov:.3f}",
            "",
            "--- MTF Vertical (Oversampled) ---",
            f"{'MTF10:': <{label_width}} {mtf10_v_ov:.3f} cycles/mm",
            f"{'MTF @ 1.0 cy/mm:': <{label_width}} {mtf1_v_ov:.3f}",
            f"{'MTF @ 2.0 cy/mm:': <{label_width}} {mtf2_v_ov:.3f}",
            f"{'MTF @ 3.0 cy/mm:': <{label_width}} {mtf3_v_ov:.3f}",
        ]

    # Save summary to txt (this line is now *outside* the if block)
    results_path = os.path.join(out_dir, "psf_results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(summary))
    print(f"Results written to {results_path}")