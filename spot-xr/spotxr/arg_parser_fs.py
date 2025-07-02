import argparse
import sys
import yaml

def get_merged_config(default_config=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=r".\fs_args.yaml", help="Path to YAML config file")

    # CLI arguments (short flags) — these will be remapped later
    parser.add_argument("--f", type=str, required=True, help="Path to the image file (.raw/.png/.tif)")
    parser.add_argument("--o", type=str, help="Output directory")
    parser.add_argument("--p", type=float, help="Pixel size in mm")
    parser.add_argument("--d", type=float, help="Circle diameter in mm")
    parser.add_argument("--no_hough", action="store_true", help="Skip Hough transform detection")
    parser.add_argument("--m", type=float, help="Magnification")
    parser.add_argument("--n", type=int, help="Minimum pixel count")
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--hl", type=int, help="Half profile length")
    parser.add_argument("--ds", type=int, help="Derivative step size")
    parser.add_argument("--axis_shifts", type=int, help="Number of axis shift steps")
    parser.add_argument("--filter", type=str, help="Reconstruction filter name")
    parser.add_argument("--sym", action="store_true", help="Symmetrize the sinogram")
    parser.add_argument("--show", action="store_true", help="Show plots")

    shift_group = parser.add_mutually_exclusive_group()
    shift_group.add_argument("--shift", dest="shift_sino", action="store_true", help="Enable sinogram shifting")
    shift_group.add_argument("--no_shift", dest="shift_sino", action="store_false", help="Disable sinogram shifting")
    parser.set_defaults(shift_sino=True)

    avg_group = parser.add_mutually_exclusive_group()
    avg_group.add_argument("--avg", dest="avg_neighbors", action="store_true", help="Enable averaging neighboring profiles")
    avg_group.add_argument("--no_avg", dest="avg_neighbors", action="store_false", help="Disable averaging neighboring profiles")
    parser.set_defaults(avg_neighbors=True)

# Parse CLI arguments, track what was passed
    args, unknown = parser.parse_known_args()
    passed_flags = {arg.split("=")[0].lstrip('-') for arg in sys.argv[1:] if arg.startswith("--")}

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cli_to_config_keys = {
        "f": "img_path",
        "o": "out_dir",
        "p": "pixel_size",
        "d": "circle_diameter",
        "m": "magnification",
        "n": "min_n",
        "nangles": "n_angles",
        "hl": "profile_half_length",
        "ds": "derivative_step",
        "axis_shifts": "axis_shifts",
        "filter": "filter_name",
        "sym": "symmetrize",
        "shift_sino": "shift_sino",
        "avg_neighbors": "avg_neighbors",
        "show": "show_plots",
    }

    cli_dict = vars(args)

    for cli_key, config_key in cli_to_config_keys.items():
        # Always overwrite config with CLI value (argparse guarantees value is set)
        config[config_key] = cli_dict[cli_key]

    return config



def validate_args(args):
    if not args["img_path"]:
        raise ValueError("Image path is required. Use --f to specify the image file.")
    if args["pixel_size"] <= 0:
        raise ValueError("Pixel size must be a positive number.")
    if args["circle_diameter"] <= 0:
        raise ValueError("Circle diameter must be a positive number.")
    if args["magnification"] is not None and args["magnification"]<= 0:
        raise ValueError("Magnification must be a positive number.")
    if args["min_n"] <= 0:
        raise ValueError("Minimum pixel count must be a positive integer.")
    if args["n_angles"] <= 0:
        raise ValueError("Number of angles must be a positive integer.")
    if args["profile_half_length"] <= 0:
        raise ValueError("Half profile length must be a positive integer.")
    if args["derivative_step"] <= 0:
        raise ValueError("Derivative step size must be a positive integer.")
    if args["axis_shifts"] < 0:
        raise ValueError("Axis shifts must be a non-negative integer.")