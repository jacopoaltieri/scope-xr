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


from . import utils, plotters
from . import arg_parser_fs as afs
from . import circle_detection as circ
from . import image_opening as io
from . import sinogram_recon as sr
from . import widths_calculator as wc


def run_pipeline_fs():
    args = afs.get_merged_config()
    afs.validate_args(args)

    print("Running focal spot reconstruction pipeline.")
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
    img_path = args["img_path"]
    pixel_size = args["pixel_size"]
    circle_diameter = args["circle_diameter"]
    no_hough = args["no_hough"]
    magnification = args["magnification"]
    min_n = args["min_n"]
    n_angles = args["n_angles"]
    profile_half_length = args["profile_half_length"]
    derivative_step = args["derivative_step"]
    axis_shifts = args["axis_shifts"]
    filter_name = args["filter_name"]
    symmetrize = args["symmetrize"]
    manual_shift = args["manual_shift"]
    auto_shift = args["auto_shift"]
    avg_neighbors = args["avg_neighbors"]
    avg_number = args["avg_number"]
    show_plots = args["show_plots"]

    # ----------------------------------------------------------------------------------#
    # Create output directory
    img_path_obj = Path(img_path)
    basename = img_path_obj.stem
    out_dir = Path(args["out_dir"]) / basename
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving outputs to {out_dir}")

    # Image loading
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

        if not hough_circle:
            raise ValueError(
                "Hough transform did not detect any circle. Provide a cropped image."
            )

        x, y, r = hough_circle
        print(f"Detected circle via Hough transform: Center=({x}, {y}), Radius={r} px")
        cropped = utils.crop_square_roi(
            img, center=(x, y), radius=r, width_factor=1.8, output_path=out_dir
        )

    cx, cy, radius = circ.estimate_circle(cropped)
    print(cx, cy, cropped.shape)
    if not circ.is_circle_centered(cropped, cx, cy):
        print("Warning: The estimated circle center is not at the image center.")
        exit(1)

    print(
        f"Estimated circle via Center Of Mass: Center=({cx}, {cy}), Radius={radius} px"
    )
    plotters.plot_circle_on_crop(cropped, cx, cy, radius, out_dir, show_plots)

    # Magnification calculation
    if magnification is not None and magnification > 0:
        m = magnification
        print(f"Using provided magnification: {m:.2f}x")
    else:
        # Compute from circle radius
        m = (radius * pixel_size) / (circle_diameter / 2)
        print(f"Estimated image magnification: {m:.2f}x")

    m_fs = m - 1  # fs magnification
    print(f"Estimated Focal Spot magnification: {m_fs:.2f}x")

    min_r = utils.eval_minimum_radius(min_n, pixel_size, m)
    if min_r > radius * pixel_size:
        print(
            f"Warning: The estimated radius {radius} mm is smaller than the minimum required radius {min_r:.2f} mm."
        )

    # Profiles and sinogram extraction
    profiles, sinogram = sr.compute_profiles_and_sinogram(
        cropped, cx, cy, radius, n_angles, profile_half_length, derivative_step
    )

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
        applied_shift = 0
        print("Sinogram shifting is disabled.")

    reconstruction = sr.reconstruct_focal_spot(sinogram, filter_name, symmetrize)

    utils.save_and_plot("profiles", profiles, out_dir)
    utils.save_and_plot("sinogram", sinogram, out_dir)
    utils.save_and_plot("reconstruction", reconstruction, out_dir)

    plotters.plot_profiles_and_reconstruction(
        profiles,
        sinogram,
        reconstruction,
        out_dir,
        show_plots,
        reconstruction_type="fs",
    )

    # Sinogram reconstruction with axis shifts
    shift_list = list(range(-axis_shifts, axis_shifts))
    shift_tiff_path = out_dir / "recon_axis_shifts.tiff"
    sr.reconstruct_with_axis_shifts(
        sinogram, shift_tiff_path, filter_name, shifts=shift_list
    )

    # Compute LSF from projection of the focal spot reconstruction
    # Horizontal LSF: sum along vertical axis (axis=0), Vertical LSF: sum along horizontal axis (axis=1)
    prof_wide_sino, prof_narrow_sino = wc.compute_lsf_from_projection(
        reconstruction)

    # Calculate FWHM and FW15M
    fw, lw, rw = wc.fwhm(prof_wide_sino)
    fn, ln, rn = wc.fwhm(prof_narrow_sino)
    print(f"Horizontal:   FWHM={fw:.2f}px")
    print(f"Vertical:     FWHM={fn:.2f}px")

    f15w, l15w, r15w = wc.fw15m(prof_wide_sino)
    f15n, l15n, r15n = wc.fw15m(prof_narrow_sino)
    print(f"Horizontal:   FW15M={f15w:.2f}px")
    print(f"Vertical:     FW15M={f15n:.2f}px")

    # Create radial coordinate array for projection profiles
    angle_step = 360.0 / n_angles
    angles = np.arange(n_angles) * angle_step
    wide_idx = np.argmin(np.abs(angles - 0))  # Closest to 0°
    narrow_idx = np.argmin(np.abs(angles - 90))  # Closest to 90°
    angle_wide_deg = angles[wide_idx]
    angle_narrow_deg = angles[narrow_idx]
    
    n_h = len(prof_wide_sino)
    radial = np.arange(n_h) - n_h // 2

    data = np.column_stack((radial, prof_wide_sino, prof_narrow_sino))

    np.savetxt(
        out_dir / "profiles.csv",
        data,
        delimiter=",",
        header="radial,horizontal_lsf,vertical_lsf",
        comments="",
        fmt=["%.6f", "%.6f", "%.6f"],
    )

    fwhm_path = out_dir / "fwhm_profiles.png"
    plotters.plot_profiles_with_fwhm(
        radial,
        prof_wide_sino,
        prof_narrow_sino,
        wide_idx,
        narrow_idx,
        0.5,
        fw,
        lw,
        rw,
        fn,
        ln,
        rn,
        fwhm_path,
        show_plots,
    )

    fw15m_path = out_dir / "fw15m_profiles.png"
    plotters.plot_profiles_with_fwhm(
        radial,
        prof_wide_sino,
        prof_narrow_sino,
        wide_idx,
        narrow_idx,
        0.15,
        f15w,
        l15w,
        r15w,
        f15n,
        l15n,
        r15n,
        fw15m_path,
        show_plots,
    )

    # Compute LSF from projection of the focal spot reconstruction
    horizontal_lsf_proj, vertical_lsf_proj = wc.compute_lsf_from_projection(
        reconstruction, 
    )

    # Calculate FWHM and FW15M for projection-based profiles
    fw_proj, lw_proj, rw_proj = wc.fwhm(horizontal_lsf_proj)
    fn_proj, ln_proj, rn_proj = wc.fwhm(vertical_lsf_proj)
    f15w_proj, l15w_proj, r15w_proj = wc.fw15m(horizontal_lsf_proj)
    f15n_proj, l15n_proj, r15n_proj = wc.fw15m(vertical_lsf_proj)

    print(
        f"Horizontal: FWHM={fw_proj:.2f}px, FW15M={f15w_proj:.2f}px"
    )
    print(
        f"Vertical:   FWHM={fn_proj:.2f}px, FW15M={f15n_proj:.2f}px"
    )

    # Save projection-based LSF profiles to file
    n_h = len(horizontal_lsf_proj)
    n_v = len(vertical_lsf_proj)
    radial_h_proj = np.arange(n_h) - n_h // 2
    radial_v_proj = np.arange(n_v) - n_v // 2

    proj_data = np.column_stack(
        (
            (
                radial_h_proj
                if len(radial_h_proj) >= len(radial_v_proj)
                else np.pad(
                    radial_h_proj,
                    (0, len(radial_v_proj) - len(radial_h_proj)),
                    constant_values=np.nan,
                )
            ),
            (
                horizontal_lsf_proj
                if len(horizontal_lsf_proj) >= len(vertical_lsf_proj)
                else np.pad(
                    horizontal_lsf_proj,
                    (0, len(vertical_lsf_proj) - len(horizontal_lsf_proj)),
                    constant_values=np.nan,
                )
            ),
            (
                radial_v_proj
                if len(radial_v_proj) >= len(radial_h_proj)
                else np.pad(
                    radial_v_proj,
                    (0, len(radial_h_proj) - len(radial_v_proj)),
                    constant_values=np.nan,
                )
            ),
            (
                vertical_lsf_proj
                if len(vertical_lsf_proj) >= len(horizontal_lsf_proj)
                else np.pad(
                    vertical_lsf_proj,
                    (0, len(horizontal_lsf_proj) - len(vertical_lsf_proj)),
                    constant_values=np.nan,
                )
            ),
        )
    )

    np.savetxt(
        out_dir / "lsf_projection.csv",
        proj_data,
        delimiter=",",
        header="radial_h,horizontal_lsf,radial_v,vertical_lsf",
        comments="",
        fmt="%.6f",
    )


    sino_with_lines_path = out_dir / "sinogram_traced_profiles.png"
    plotters.plot_sinogram_with_traced_profiles(
        sinogram,
        wide_idx,
        narrow_idx,
        sino_with_lines_path,
        reconstruction_type="fs",
        show_plots=show_plots,
    )

    spot_with_lines_path = out_dir / "focal_spot_traced_profiles.png"
    plotters.plot_recon_with_lines(
        reconstruction,
        angle_wide_deg,
        angle_narrow_deg,
        out_path=spot_with_lines_path,
        show_plots=show_plots,
        reconstruction_type="fs",
    )

    # Compute focal spot physical dimensions
    wide_fs = wc.compute_fs_width(fw, pixel_size, m_fs)
    narrow_fs = wc.compute_fs_width(fn, pixel_size, m_fs)
    print(f"Horizontal focal spot width (FWHM): {wide_fs:.3f} mm")
    print(f"Vertical focal spot width (FWHM):   {narrow_fs:.3f} mm")
    wide_fs15m = wc.compute_fs_width(f15w, pixel_size, m_fs)
    narrow_fs15m = wc.compute_fs_width(f15n, pixel_size, m_fs)
    print(f"Horizontal focal spot width (FW15M): {wide_fs15m:.3f} mm")
    print(f"Vertical focal spot width (FW15M):   {narrow_fs15m:.3f} mm")

    # wide_fs_erf = wc.compute_fs_width(fw_erf, pixel_size, m_fs)
    # narrow_fs_erf = wc.compute_fs_width(fn_erf, pixel_size, m_fs)
    # print(f"Widest focal spot width (ERF): {wide_fs_erf:.3f} mm")
    # print(f"Narrowest focal spot width (ERF): {narrow_fs_erf:.3f} mm")

    # Create results summary
    label_width = 24

    summary = [
        "========================================",
        "  SCOPE-XR Focal Spot Analysis Results",
        "========================================",
        "",
        f"Full arguments: {args}",  # For traceability
        "--- General Info ---",
        f"{'Output Directory:': <{label_width}} {out_dir}",
        "",
        "--- Setup Parameters ---",
        f"{'COM Circle Center:': <{label_width}} ({cx:.2f}, {cy:.2f}) px",
        f"{'COM Circle Radius:': <{label_width}} {radius:.2f} px",
        f"{'Image Magnification:': <{label_width}} {m:.2f}x",
        f"{'Focal Spot Mag:': <{label_width}} {m_fs:.2f}x",
        f"{'Applied Axis Shift:': <{label_width}} {applied_shift} px (Search Range: +/-{axis_shifts})",
        "",
        # --- Group by Widest Profile ---
        "--- Horizontal Profile Results ---",
        f"{'FWHM:': <{label_width}} {fw:.3f} px",
        f"{'FW15M:': <{label_width}} {f15w:.3f} px",
        f"{'Spot Size (FWHM):': <{label_width}} {wide_fs:.3f} mm",
        f"{'Spot Size (FW15M):': <{label_width}} {wide_fs15m:.3f} mm",
        "",
        # --- Group by Narrowest Profile ---
        "--- Vertical Profile Results ---",
        f"{'FWHM:': <{label_width}} {fn:.3f} px",
        f"{'FW15M:': <{label_width}} {f15n:.3f} px",
        f"{'Spot Size (FWHM):': <{label_width}} {narrow_fs:.3f} mm",
        f"{'Spot Size (FW15M):': <{label_width}} {narrow_fs15m:.3f} mm",
        "",
    ]

    # Save summary to .txt
    results_path = out_dir / "fs_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"Results written to {results_path}")
