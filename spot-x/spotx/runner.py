import os
import numpy as np
import matplotlib.pyplot as plt

from spotx import utils, plotters
from spotx import image_opening as io
from spotx import circle_detection as circ
from spotx import sinogram_recon as sr
from spotx import widths_calculator as wc
from spotx.arg_parser import get_merged_config, validate_args


def run_pipeline():
    args = get_merged_config()
    validate_args(args)

    print("Arguments in use:")
    for k, v in args.items():
        if k != "hough_params":
            print(f"  {k:18}: {v}")

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
    shift_sino = args["shift_sino"]
    avg_neighbors = args["avg_neighbors"]
    show_plots = args["show_plots"]

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

    # circle detetion
    if no_hough:
        print(
            "Caution! Hough transform not used. Using provided image as already cropped."
        )
        cropped = img
    else:
        # Detect circle using Hough Transform
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

    # Estimate magnification
    if magnification is not None:
        m = magnification
        print(f"Using provided magnification: {m:.2f}x")
    else:
        # compute from circle radius
        m = (radius * pixel_size) / (circle_diameter / 2)
        print(f"Estimated image magnification: {m:.2f}x")

    m_fs = m - 1  # fs magnification
    print(f"Estimated fs magnification: {m_fs:.2f}x")

    min_r = utils.eval_minimum_radius(min_n, pixel_size, m)
    if min_r > radius*pixel_size:
        print(
            f"Warning: The estimated radius {radius} mm is smaller than the minimum required radius {min_r:.2f} mm."
        )

    # Extract profiles and sinogram
    profiles, sinogram = sr.compute_profiles_and_sinogram(
        cropped, cx, cy, radius, n_angles, profile_half_length, derivative_step
    )

    if shift_sino:
        centered_sino, applied_shift = sr.auto_center_sinogram(sinogram)
        if applied_shift == 0:
            sinogram = centered_sino
        else:
            sinogram = centered_sino[applied_shift:-applied_shift, :]
        print(f"Applied axis shift: {applied_shift} px")

    reconstruction = sr.reconstruct_focal_spot(sinogram, filter_name, symmetrize)

    # Save images
    saved_files = []
    for name, arr in [
        ("profiles.png", profiles),
        ("sinogram.png", sinogram),
        ("reconstruction.png", reconstruction),
    ]:
        path = os.path.join(out_dir, name)
        plt.imsave(path, arr, cmap="gray")
        saved_files.append(path)

    plotters.plot_profiles_and_reconstruction(
        profiles, sinogram, reconstruction, out_dir, show_plots
    )

    # Shift the central axis and save as a sequence. This is useful to see if the centering is correct.
    shift_list = list(range(-axis_shifts,axis_shifts))
    shift_tiff_path = os.path.join(out_dir, "recon_axis_shifts.tiff")
    sr.reconstruct_with_axis_shifts(
        sinogram, shift_tiff_path, filter_name, shifts=shift_list
    )

    wide_idx, narrow_idx, sigmas = wc.find_extreme_profiles_erf(profiles)
    print(f"Widest edge at angle idx {wide_idx}")
    print(f"Narrowest edge at angle idx {narrow_idx}")

    if avg_neighbors:
        prof_wide_sino = wc.average_neighbors(sinogram, wide_idx)
        prof_narrow_sino = wc.average_neighbors(sinogram, narrow_idx)
    else:
        prof_wide_sino = sinogram[:, wide_idx]
        prof_narrow_sino = sinogram[:, narrow_idx]

    fw, lw, rw = wc.fwhm(prof_wide_sino)
    fn, ln, rn = wc.fwhm(prof_narrow_sino)
    print(f"Widest:   FWHM={fw}px (from {lw} to {rw})")
    print(f"Narrowest: FWHM={fn}px (from {ln} to {rn})")

    fw_erf = wc.fwhm_from_erf_sigma(sigmas[wide_idx])
    fn_erf = wc.fwhm_from_erf_sigma(sigmas[narrow_idx])
    print(f"Widest (ERF):   FWHM={fw_erf:.2f}px")
    print(f"Narrowest (ERF): FWHM={fn_erf:.2f}px")

    n_rays = sinogram.shape[0]
    radial = np.arange(n_rays) - n_rays // 2

    fwhm_path = os.path.join(out_dir, "fwhm_sinogram_profiles.png")
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

    sino_with_lines_path = os.path.join(out_dir, "sinogram_traced_profiles.png")
    plotters.plot_sinogram_with_traced_profiles(
        sinogram, wide_idx, narrow_idx, sino_with_lines_path, show_plots
    )

    angle_step = 360.0 / n_angles
    angle_wide_deg = wide_idx * angle_step
    angle_narrow_deg = narrow_idx * angle_step
    spot_with_lines_path = os.path.join(out_dir, "focal_spot_traced_profiles.png")
    plotters.plot_focal_spot_with_lines(
        reconstruction,
        angle_wide_deg,
        angle_narrow_deg,
        out_path=spot_with_lines_path,
        show_plots=show_plots,
    )

    wide_fs = wc.compute_fs_width(fw, pixel_size, m_fs)
    narrow_fs = wc.compute_fs_width(fn, pixel_size, m_fs)
    print(f"Widest focal spot width: {wide_fs:.3f} mm")
    print(f"Narrowest focal spot width: {narrow_fs:.3f} mm")

    wide_fs_erf = wc.compute_fs_width(fw_erf, pixel_size, m_fs)
    narrow_fs_erf = wc.compute_fs_width(fn_erf, pixel_size, m_fs)
    print(f"Widest focal spot width (ERF): {wide_fs_erf:.3f} mm")
    print(f"Narrowest focal spot width (ERF): {narrow_fs_erf:.3f} mm")

    # Create results summary
    summary = [
        f"Output saved to: {out_dir}",
        f"Arguments: {args}",
        f"COM circle: center=({cx},{cy}), radius={radius}px",
        f"Magnification: image={m:.2f}x, focal spot={m_fs:.2f}x",
        f"Applied shift: {applied_shift}px ({axis_shifts})",
        f"FWHM classic: widest={fw}px (idx {wide_idx}), narrowest={fn}px (idx {narrow_idx})",
        f"FWHM ERF:     widest={fw_erf:.2f}px, narrowest={fn_erf:.2f}px",
        f"Spot size mm classic: widest={wide_fs:.2f}, narrowest={narrow_fs:.2f}",
        f"Spot size mm ERF:     widest={wide_fs_erf:.3f}, narrowest={narrow_fs_erf:.3f}",
        f"Angles: wide={wide_idx*angle_step:.1f}°, narrow={narrow_idx*angle_step:.1f}°",
    ]

    # Save summary to txt
    results_path = os.path.join(out_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(summary))
    print(f"Results written to {results_path}")
