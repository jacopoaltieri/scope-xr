import argparse
import sys
import yaml

def get_merged_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=r".\psf_args.yaml", help="Path to YAML config file")

    # CLI arguments
    parser.add_argument("--f", type=str, required=True, help="Path to the image file (.raw/.png/.tif)")
    parser.add_argument("--o", type=str, help="Output directory")
    parser.add_argument("--p", type=float, help="Pixel size in mm")
    parser.add_argument("--d", type=float, help="Circle diameter in mm")
    parser.add_argument("--no_hough", action="store_true", help="Skip Hough transform detection")
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--hl", type=int, help="Half profile length")
    parser.add_argument("--ds", type=int, help="Derivative step size")
    parser.add_argument("--filter", type=str, help="Reconstruction filter name")
    parser.add_argument("--sym", action="store_true", help="Symmetrize the sinogram")
    parser.add_argument("--dtheta", type=float, help="Angle of circular sector for oversampling in degrees")
    parser.add_argument("--resample1", type=float, help="First resample factor (fine grid).")
    parser.add_argument("--resample2", type=float, help="Second resample factor (coarse grid). This will be the final oversampling factor.")
    parser.add_argument("--gaussian_sigma", type=float, help=" Standard deviation of the gaussian blur applied between the fine and the coarse resampling.")
    parser.add_argument("--show", action="store_true", help="Show plots")

    shift_group = parser.add_mutually_exclusive_group()
    shift_group.add_argument("--shift", dest="shift_sino", action="store_true", help="Enable sinogram shifting")
    shift_group.add_argument("--no_shift", dest="shift_sino", action="store_false", help="Disable sinogram shifting")
    parser.set_defaults(shift_sino=True)

    avg_group = parser.add_mutually_exclusive_group()
    avg_group.add_argument("--avg", dest="avg_neighbors", action="store_true", help="Enable averaging neighboring profiles")
    avg_group.add_argument("--no_avg", dest="avg_neighbors", action="store_false", help="Disable averaging neighboring profiles")
    parser.set_defaults(avg_neighbors=True)

    oversample_group = parser.add_mutually_exclusive_group()
    oversample_group.add_argument("--oversample",dest="oversample", action="store_true", help="Enable oversampling")
    oversample_group.add_argument("--no_oversample",dest="oversample", action="store_false", help="Enable oversampling")
    parser.set_defaults(oversample=True)


    args, unknown = parser.parse_known_args()
    passed_flags = {arg.split("=")[0].lstrip('-') for arg in sys.argv[1:] if arg.startswith("--")}

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI to config key mapping
    cli_to_config_keys = {
        "f": "img_path",
        "o": "out_dir",
        "p": "pixel_size",
        "d": "circle_diameter",
        "no_hough": "no_hough",
        "nangles": "n_angles",
        "hl": "profile_half_length",
        "ds": "derivative_step",
        "filter": "filter_name",
        "sym": "symmetrize",
        "shift_sino": "shift_sino",
        "show": "show_plots",
        "avg_neighbors": "avg_neighbors",
        "oversample": "oversample",
        "dtheta": "dtheta",
        "resample1": "resample1",
        "resample2": "resample2",
        "gaussian_sigma": "gaussian_sigma",
    }

    cli_dict = vars(args)

    for cli_key, config_key in cli_to_config_keys.items():
        # Only override if the value was explicitly set in CLI
        # We can check if it differs from the YAML value, or always prioritize CLI when not None
        if cli_dict[cli_key] is not None:
            config[config_key] = cli_dict[cli_key]


    return config


def validate_args(args):
    if not args["img_path"]:
        raise ValueError("Image path is required. Use --f to specify the image file.")
    if args["pixel_size"] <= 0:
        raise ValueError("Pixel size must be a positive number.")
    if args["circle_diameter"] <= 0:
        raise ValueError("Circle diameter must be a positive number.")
    if args["n_angles"] <= 0:
        raise ValueError("Number of angles must be a positive integer.")
    if args["profile_half_length"] <= 0:
        raise ValueError("Half profile length must be a positive integer.")
    if args["derivative_step"] <= 0:
        raise ValueError("Derivative step size must be a positive integer.")
    if "dtheta" in args and args["dtheta"] <= 0:
        raise ValueError("dtheta must be a positive number.")
    if "resample1" in args and args["resample1"] is not None and args["resample1"] <= 0:
        raise ValueError("resample1 must be a positive number.")
    if "resample2" in args and args["resample2"] is not None and args["resample2"] <= 0:
        raise ValueError("resample2 must be a positive number.")
    if "gaussian_sigma" in args and args["gaussian_sigma"] is not None and args["gaussian_sigma"] < 0:
        raise ValueError("gaussian_sigma must be non-negative.")
