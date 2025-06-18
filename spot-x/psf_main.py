import os
import numpy as np
import matplotlib.pyplot as plt

from psf.arg_parser import get_merged_config, validate_args
from spotx import image_opening as io
from spotx import circle_detection as circ
from spotx import utils, plotters
from spotx import sinogram_recon as sr


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
