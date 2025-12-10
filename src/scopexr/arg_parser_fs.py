import argparse
import sys
import yaml


def get_merged_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default=r".\fs_args.yaml", help="Path to YAML config file"
    )

    # CLI arguments (short flags)
    parser.add_argument("--f", type=str, help="Path to the image file (.raw/.png/.tif)")
    parser.add_argument("--o", type=str, help="Output directory")
    parser.add_argument("--p", type=float, help="Pixel size in mm")
    parser.add_argument("--d", type=float, help="Circle diameter in mm")
    parser.add_argument(
        "--no_hough",
        action="store_true",
        default=None,
        help="Skip Hough transform detection",
    )
    parser.add_argument("--m", type=float, help="Magnification")
    parser.add_argument("--n", type=int, help="Minimum pixel count")
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--hl", type=int, help="Half profile length")
    parser.add_argument("--ds", type=int, help="Derivative step size")
    parser.add_argument("--axis_shifts", type=int, help="Number of axis shift steps")
    parser.add_argument("--filter", type=str, help="Reconstruction filter name")
    parser.add_argument("--avg_number", type=int, help="Number of profiles to average")
    parser.add_argument(
        "--sym", action="store_true", default=None, help="Symmetrize the sinogram"
    )
    parser.add_argument("--show", action="store_true", default=None, help="Show plots")

    shift_group = parser.add_mutually_exclusive_group()
    shift_group.add_argument(
        "--auto_shift",
        action="store_true",
        default=None,
        help="Enable automatic sinogram centering.",
    )
    shift_group.add_argument(
        "--manual_shift",
        type=int,
        default=None,
        help="Provide a specific manual shift value (in pixels).",
    )
    shift_group.add_argument(
        "--no_shift",
        action="store_true",
        default=None,
        help="Disable all sinogram shifting.",
    )

    avg_group = parser.add_mutually_exclusive_group()
    avg_group.add_argument(
        "--avg",
        dest="avg_neighbors",
        action="store_true",
        default=None,  # Use None as default
        help="Enable averaging neighboring profiles",
    )
    avg_group.add_argument(
        "--no_avg",
        dest="avg_neighbors",
        action="store_false",
        default=None,  # Use None as default
        help="Disable averaging neighboring profiles",
    )

    args, unknown = parser.parse_known_args()

    # 1. Set code defaults (lowest priority)
    config = {
        "img_path": None,
        "auto_shift": True,
        "manual_shift": None,
        "no_shift": False,
        "avg_neighbors": False,
        "no_hough": False,
        "symmetrize": False,
        "show_plots": False,
    }

    # 2. Load YAML config (overwrites code defaults)
    try:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
    except FileNotFoundError:
        print(
            f"Warning: Config file not found at {args.config}. Using defaults.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Error loading YAML config: {e}", file=sys.stderr)

    # 3. Load CLI arguments (highest priority)
    cli_to_config_keys = {
        "f": "img_path",
        "o": "out_dir",
        "p": "pixel_size",
        "d": "circle_diameter",
        "no_hough": "no_hough",
        "m": "magnification",
        "n": "min_n",
        "nangles": "n_angles",
        "hl": "profile_half_length",
        "ds": "derivative_step",
        "axis_shifts": "axis_shifts",
        "filter": "filter_name",
        "sym": "symmetrize",
        "avg_neighbors": "avg_neighbors",
        "avg_number": "avg_number",
        "show": "show_plots",
        # Add all shift keys to the map
        "auto_shift": "auto_shift",
        "manual_shift": "manual_shift",
        "no_shift": "no_shift",
    }

    cli_dict = vars(args)

    for cli_key, config_key in cli_to_config_keys.items():
        cli_value = cli_dict.get(cli_key)
        # Only update if the CLI argument was *actually* given
        if cli_value is not None:
            config[config_key] = cli_value

    # ---
    # Check which CLI flag was *actually passed* (from cli_dict)
    # and enforce priority.
    # ---
    if cli_dict.get("manual_shift") is not None:
        # CLI --manual_shift was used
        config["auto_shift"] = False
        config["no_shift"] = False
        config["manual_shift"] = cli_dict.get("manual_shift")
    elif cli_dict.get("auto_shift") is True:
        # CLI --auto_shift was used
        config["auto_shift"] = True
        config["no_shift"] = False
        config["manual_shift"] = None
    elif cli_dict.get("no_shift") is True:
        # CLI --no_shift was used
        config["auto_shift"] = False
        config["no_shift"] = True
        config["manual_shift"] = None
    # If no CLI shift flag was passed, the config (from YAML or default) is used as-is.

    return config


def validate_args(args):
    if not args.get("img_path"):
        raise ValueError("Image path is required. Use --f to specify the image file.")
    if args.get("pixel_size") is None or args["pixel_size"] <= 0:
        raise ValueError("Pixel size must be a positive number.")
    if args.get("circle_diameter") is None or args["circle_diameter"] <= 0:
        raise ValueError("Circle diameter must be a positive number.")
    if args.get("magnification") is not None and args["magnification"] <= 0:
        raise ValueError("Magnification must be a positive number.")
    if args.get("min_n") is None or args["min_n"] <= 0:
        raise ValueError("Minimum pixel count must be a positive integer.")
    if args.get("n_angles") is None or args["n_angles"] <= 0:
        raise ValueError("Number of angles must be a positive integer.")
    if args.get("profile_half_length") is None or args["profile_half_length"] <= 0:
        raise ValueError("Half profile length must be a positive integer.")
    if args.get("derivative_step") is None or args["derivative_step"] <= 0:
        raise ValueError("Derivative step size must be a positive integer.")
    if args.get("axis_shifts") is None or args["axis_shifts"] < 0:
        raise ValueError("Axis shifts must be a non-negative integer.")

    avg_num = args.get("avg_number")
    if args.get("avg_neighbors") and avg_num is not None:
        if avg_num <= 0 or avg_num % 2 == 0:
            raise ValueError("Average number must be a positive odd integer.")

    if args.get("manual_shift") is not None and not isinstance(
        args["manual_shift"], int
    ):
        raise ValueError("Manual shift must be an integer.")
