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
from scipy.optimize import curve_fit


from . import utils, plotters
from . import arg_parser_psf as apsf
from . import circle_detection as circ
from . import image_opening as io
from . import mtf_calc as mtfc
from . import sinogram_extraction as se
from . import sinogram_reconstruction as srec
from . import widths_calculator as wc


def run_pipeline_psf():
    args = apsf.get_merged_config()
    apsf.validate_args(args)

    print("Running detector PSF reconstruction pipeline.")
    print("Arguments in use:")
    for k, v in args.items():
        if k != "hough_params":
            print(f"  {k:18}: {v}")
    hough_params = args.get("hough_params", {})
    if hough_params:
        print("Hough params:")
        for key in (
            "dp",
            "min_dist",
            "param1",
            "param2",
            "min_radius",
            "max_radius",
            "debug",
        ):
            if key in hough_params:
                print(f"  {key:18}: {hough_params[key]}")

    # ----------------------------------------------------------------------------------#
    # Extract configuration parameters
    img_path = args["img_path"]
    pixel_size = args["pixel_size"]
    no_hough = args["no_hough"]
    n_angles = args["n_angles"]
    profile_half_length = args["profile_half_length"]
    derivative_step = args["derivative_step"]
    axis_shifts = args["axis_shifts"]
    filter_name = args["filter_name"]
    symmetrize = args["symmetrize"]
    manual_shift = args["manual_shift"]
    auto_shift = args["auto_shift"]
    oversample = args["oversample"]
    dtheta = args["dtheta"]
    resample2 = args["resample2"]
    gaussian_sigma = args["gaussian_sigma"]
    show_plots = args["show_plots"]

    # ----------------------------------------------------------------------------------#
    # Create output directory
    img_path_obj = Path(img_path)
    basename = img_path_obj.stem
    out_dir = Path(args.get("out_dir", ".")) / basename
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {out_dir}")

    # Load image
    try:
        img = io.load_image(img_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to load image at `{img_path}`: {e}") from e

    # Circle detection
    if no_hough:
        print(
            "Caution! Hough transform not used. Using provided image as already cropped."
        )
        cropped = img
    else:
        hough_params = args["hough_params"]
        hough_circle = circ.detect_circle_hough(
            img,
            dp=hough_params["dp"],
            min_dist=hough_params["min_dist"],
            param1=hough_params["param1"],
            param2=hough_params["param2"],
            min_radius=hough_params["min_radius"],
            max_radius=hough_params["max_radius"],
            output_path=out_dir,
            debug=hough_params.get("debug", False),
        )

        x, y, r = hough_circle
        print(f"Detected circle via Hough transform: Center=({x}, {y}), Radius={r} px")
        cropped = utils.crop_square_roi(
            img, center=(x, y), radius=r, width_factor=1.2, output_path=out_dir
        )

        if not hough_circle:
            raise ValueError(
                "Hough transform did not detect any circle. Provide a cropped image."
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
    profiles, sinogram = se.compute_profiles_and_sinogram(
        cropped, cx, cy, radius, n_angles, profile_half_length, derivative_step
    )

    if manual_shift is not None:
        print(f"Applying manual shift: {manual_shift} px")
        centered_sino, applied_shift = se.manual_center_sinogram(sinogram, manual_shift)
        sinogram = centered_sino
    elif auto_shift:
        print("Running automatic sinogram centering...")
        centered_sino, applied_shift = se.auto_center_sinogram(sinogram)
        sinogram = centered_sino
        print(f"Applied automatic axis shift: {applied_shift} px")
    else:
        applied_shift = 0
        print("Sinogram shifting is disabled.")

    reconstruction = srec.reconstruct_focal_spot(sinogram, filter_name, symmetrize)

    utils.save_and_plot("profiles", profiles, out_dir)
    utils.save_and_plot("sinogram", sinogram, out_dir)
    utils.save_and_plot("reconstruction", reconstruction, out_dir)

    plotters.plot_profiles_and_reconstruction(
        profiles,
        sinogram,
        reconstruction,
        out_dir,
        show_plots,
        reconstruction_type="psf",
    )

    # Sinogram reconstruction with axis shifts
    shift_list = list(range(-axis_shifts, axis_shifts))
    shift_tiff_path = out_dir / "recon_axis_shifts.tiff"
    srec.reconstruct_with_axis_shifts(
        sinogram,
        shift_tiff_path,
        filter_name,
        shift_list,
        symmetrize,
    )

    # Compute LSF from projection of the focal spot reconstruction
    # Horizontal LSF: sum along vertical axis (axis=0), Vertical LSF: sum along horizontal axis (axis=1)
    prof_horizontal, prof_vertical = wc.compute_lsf_from_projection(reconstruction)

    # Calculate FWHM
    fh, lh, rh = wc.fw_at_percent_max(prof_horizontal, 0.5)
    fv, lv, rv = wc.fw_at_percent_max(prof_vertical, 0.5)
    print(f"Horizontal:   FWHM={fh:.2f}px")
    print(f"Vertical:     FWHM={fv:.2f}px")

    # Create radial coordinate array for projection profiles
    angle_step = 360.0 / n_angles
    angles = np.arange(n_angles) * angle_step
    horizontal_idx = np.argmin(np.abs(angles - 0))  # Closest to 0°
    vertical_idx = np.argmin(np.abs(angles - 90))  # Closest to 90°

    radial = np.arange(len(prof_horizontal)) - (len(prof_horizontal) // 2)
    data = np.column_stack((radial, prof_horizontal, prof_vertical))

    # np.savetxt(
    #     out_dir / "profiles.csv",
    #     data,
    #     delimiter=",",
    #     header="radial,horizontal_lsf,vertical_lsf",
    #     comments="",
    #     fmt=["%.6f", "%.6f", "%.6f"],
    # )

    # Fit Gaussian to projection-based profiles for plotting
    def gaussian_fit(x, A, mu, sigma, B):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + B

    try:
        popt_h_corr, _ = curve_fit(
            gaussian_fit,
            radial,
            prof_horizontal,
            p0=[np.max(prof_horizontal), 0, fh / 2.355, 0],
            maxfev=5000,
        )

    except Exception:
        popt_h_corr = np.array([np.max(prof_horizontal), 0, fh / 2.355, 0])

    try:
        popt_v_corr, _ = curve_fit(
            gaussian_fit,
            radial,
            prof_vertical,
            p0=[np.max(prof_vertical), 0, fv / 2.355, 0],
            maxfev=5000,
        )
    except Exception:
        popt_v_corr = np.array([np.max(prof_vertical), 0, fv / 2.355, 0])

    # Plot profiles with Gaussian fits
    plotters.plot_profile_with_gaussian(
        radial=radial,
        intensity_profile=prof_horizontal,
        popt=popt_h_corr,
        out_path=out_dir / "lsf_profile_horizontal.png",
        show_plots=show_plots,
        pixel_size=pixel_size,
        magnification=1.0,
    )

    plotters.plot_profile_with_gaussian(
        radial=radial,
        intensity_profile=prof_vertical,
        popt=popt_v_corr,
        out_path=out_dir / "lsf_profile_vertical.png",
        show_plots=show_plots,
        pixel_size=pixel_size,
        magnification=1.0,
    )

    # Plot sinogram and reconstruction with lines
    plotters.plot_sinogram_with_traced_profiles(
        sinogram,
        horizontal_idx,
        vertical_idx,
        out_dir / "sinogram_traced_profiles.png",
        reconstruction_type="psf",
        show_plots=show_plots,
    )
    plotters.plot_recon_with_lines(
        reconstruction,
        horizontal_idx,
        vertical_idx,
        out_dir / "psf_traced_profiles.png",
        show_plots=show_plots,
        reconstruction_type="psf",
    )

    # Compute MTF in horizontal and vertical directions

    # Compute MTF from the projection-based LSF profiles
    freq_h, mtf_h, mtf10_h = mtfc.compute_1d_mtf(prof_horizontal, pixel_size)

    mtf1_h = mtfc.get_mtf_at_freq(1.0, freq_h, mtf_h)
    mtf2_h = mtfc.get_mtf_at_freq(2.0, freq_h, mtf_h)
    mtf3_h = mtfc.get_mtf_at_freq(3.0, freq_h, mtf_h)

    freq_v, mtf_v, mtf10_v = mtfc.compute_1d_mtf(
        prof_vertical,
        pixel_size,
    )

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
        out_path=out_dir / "mtf_horizontal.png",
        mtf10_freq=mtf10_h,
        show_plots=show_plots,
    )
    plotters.plot_1d_mtf(
        freq_v,
        mtf_v,
        pixel_size=pixel_size,
        out_path=out_dir / "mtf_vertical.png",
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
        f"{'Input Image:': <{label_width}} {Path(img_path).name}",
        f"{'Output Directory:': <{label_width}} {out_dir}",
        "",
        "--- Setup Parameters ---",
        f"{'COM Circle Center:': <{label_width}} ({cx:.2f}, {cy:.2f}) px",
        f"{'COM Circle Radius:': <{label_width}} {radius:.2f} px",
        f"{'LSF Method:': <{label_width}} Projection-based",
        "",
        "--- PSF Size (FWHM from Projection) ---",
        f"{'FWHM Horizontal:': <{label_width}} {fh:.3f} px",
        f"{'FWHM Vertical:': <{label_width}} {fv:.3f} px",
        "",
        "--- MTF Horizontal (from Projection) ---",
        f"{'MTF10:': <{label_width}} {mtf10_h:.3f} cycles/mm",
        f"{'MTF @ 1.0 cy/mm:': <{label_width}} {mtf1_h:.3f}",
        f"{'MTF @ 2.0 cy/mm:': <{label_width}} {mtf2_h:.3f}",
        f"{'MTF @ 3.0 cy/mm:': <{label_width}} {mtf3_h:.3f}",
        "",
        "--- MTF Vertical (from Projection) ---",
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

        # Treat None the same as 0 for gaussian_sigma
        if gaussian_sigma is None or gaussian_sigma == 0.0:
            gaussian_sigma = 0.0
            print("Using oversampling with binned statistics (no Gaussian blur).")
        else:
            print(
                f"Using oversampling with 3-step Gaussian blur (sigma={gaussian_sigma})."
            )

        profiles_oversampled, sinogram_oversampled = (
            se.compute_subpixel_profiles_and_sinogram(
                cropped,
                cx,
                cy,
                radius,
                n_angles,
                profile_half_length,
                derivative_step,
                dtheta,
                resample=resample2,
                gaussian_sigma=gaussian_sigma,
            )
        )

        applied_shift_ov = 0  # New variable for oversampled shift
        if manual_shift is not None:
            # Scale the manual shift (which is in 'normal' pixels)
            manual_shift_ov = int(manual_shift * resample2)
            print(
                f"Applying manual shift to oversampled sinogram: {manual_shift_ov} px"
            )
            # Apply shift to sinogram_oversampled
            centered_sino, applied_shift_ov = se.manual_center_sinogram(
                sinogram_oversampled, manual_shift_ov
            )
            sinogram_oversampled = centered_sino
        elif auto_shift:
            print("Running automatic sinogram centering (oversampled)...")
            # Apply auto-shift to sinogram_oversampled
            centered_sino, applied_shift_ov = se.auto_center_sinogram(
                sinogram_oversampled
            )
            sinogram_oversampled = centered_sino
            print(f"Applied automatic axis shift: {applied_shift_ov} px (oversampled)")
        else:
            # No shift is applied
            applied_shift_ov = 0
            print("Sinogram shifting is disabled (oversampled).")

        recon_oversampled = srec.reconstruct_focal_spot(
            sinogram_oversampled, filter_name, symmetrize
        )

        utils.save_and_plot("profiles_oversampled", profiles_oversampled, out_dir)
        utils.save_and_plot("sinogram_oversampled", sinogram_oversampled, out_dir)
        utils.save_and_plot("reconstruction_oversampled", recon_oversampled, out_dir)

        plotters.plot_profiles_and_reconstruction(
            profiles_oversampled,
            sinogram_oversampled,
            recon_oversampled,
            out_dir,
            show_plots,
            reconstruction_type="psf",
            suffix="_oversampled",
        )

        # Oversampled sinogram reconstruction with axis shifts
        shift_list = list(range(-axis_shifts, axis_shifts))
        shift_tiff_path = out_dir / "oversampled_recon_axis_shifts.tiff"
        srec.reconstruct_with_axis_shifts(
            sinogram_oversampled,
            shift_tiff_path,
            filter_name,
            shift_list,
            symmetrize,
        )

        # Compute LSF from projection of the oversampled PSF reconstruction
        prof_h_ov, prof_v_ov = wc.compute_lsf_from_projection(recon_oversampled)

        # FWHM value from oversampled (in 'oversampled pixels')
        fw_h_ov_native, _, _ = wc.fw_at_percent_max(prof_h_ov, 0.5)
        fw_v_ov_native, _, _ = wc.fw_at_percent_max(prof_v_ov, 0.5)
        # Convert FWHM to 'normal' pixel-equivalent
        fw_h_ov = fw_h_ov_native / resample2
        fw_v_ov = fw_v_ov_native / resample2

        print(f"Horizontal (Oversampled):  FWHM={fw_h_ov:.2f} px")
        print(f"Vertical (Oversampled):    FWHM={fw_v_ov:.2f} px")

        # The radial axis for oversampled plot
        radial_ov = np.arange(len(prof_h_ov)) - (len(prof_h_ov) // 2)

        data_oversampled = np.column_stack((radial_ov, prof_h_ov, prof_v_ov))

        # np.savetxt(
        #     out_dir / "profiles_oversampled.csv",
        #     data_oversampled,
        #     delimiter=",",
        #     header="radial,horizontal_lsf,vertical_lsf",
        #     comments="",
        #     fmt=["%.6f", "%.6f", "%.6f"],
        # )

        def gaussian_fit(x, A, mu, sigma, B):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + B

        try:
            popt_h_ov, _ = curve_fit(
                gaussian_fit,
                radial_ov,
                prof_h_ov,
                p0=[np.max(prof_h_ov), 0, fw_h_ov / 2.355, 0],
                maxfev=5000,
            )

        except Exception:
            popt_h_ov = np.array([np.max(prof_h_ov), 0, fw_h_ov / 2.355, 0])

        try:
            popt_v_ov, _ = curve_fit(
                gaussian_fit,
                radial_ov,
                prof_v_ov,
                p0=[np.max(prof_v_ov), 0, fw_v_ov / 2.355, 0],
                maxfev=5000,
            )
        except Exception:
            popt_v_ov = np.array([np.max(prof_v_ov), 0, fw_v_ov / 2.355, 0])

        plotters.plot_profile_with_gaussian(
            radial=radial_ov,
            intensity_profile=prof_h_ov,
            popt=popt_h_ov,
            out_path=out_dir / "oversampled_lsf_profile_horizontal.png",
            show_plots=show_plots,
            pixel_size=pixel_size,  # Plot against oversampled pixel size
            magnification=1.0,
        )
        plotters.plot_profile_with_gaussian(
            radial=radial_ov,
            intensity_profile=prof_v_ov,
            popt=popt_v_ov,
            out_path=out_dir / "oversampled_lsf_profile_vertical.png",
            show_plots=show_plots,
            pixel_size=pixel_size,  # Plot against oversampled pixel size
            magnification=1.0,
        )

        plotters.plot_sinogram_with_traced_profiles(
            sinogram_oversampled,
            horizontal_idx,
            vertical_idx,
            out_dir / "oversampled_sinogram_traced_profiles.png",
            reconstruction_type="psf",
            show_plots=show_plots,
        )
        plotters.plot_recon_with_lines(
            recon_oversampled,
            horizontal_idx,
            vertical_idx,
            out_dir / "psf_traced_profiles_oversampled.png",
            show_plots=show_plots,
            reconstruction_type="psf",
        )

        # Compute MTF from projection-based LSF profiles
        freq_h_ov, mtf_h_ov, mtf10_h_ov = mtfc.compute_1d_mtf(
            prof_h_ov, pixel_size / resample2
        )

        mtf1_h_ov = mtfc.get_mtf_at_freq(1.0, freq_h_ov, mtf_h_ov)
        mtf2_h_ov = mtfc.get_mtf_at_freq(2.0, freq_h_ov, mtf_h_ov)
        mtf3_h_ov = mtfc.get_mtf_at_freq(3.0, freq_h_ov, mtf_h_ov)

        freq_v_ov, mtf_v_ov, mtf10_v_ov = mtfc.compute_1d_mtf(
            prof_v_ov, pixel_size / resample2
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
            pixel_size=pixel_size,  # Plot against original Nyquist
            out_path=out_dir / "mtf_horizontal_oversampled.png",
            mtf10_freq=mtf10_h_ov,
            show_plots=show_plots,
        )
        plotters.plot_1d_mtf(
            freq_v_ov,
            mtf_v_ov,
            pixel_size=pixel_size,  # Plot against original Nyquist
            out_path=out_dir / "mtf_vertical_oversampled.png",
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

    # Save summary to txt
    results_path = out_dir / "psf_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"Results written to {results_path}")
