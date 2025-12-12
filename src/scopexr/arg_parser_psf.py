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

import argparse
import sys
import yaml


def get_merged_config() -> dict:
    """
    Parse command-line arguments and YAML configuration file,
    merging them with the following priority:
    
    1. Code defaults (lowest priority)
    2. YAML config file
    3. Command-line arguments (highest priority)

    Returns
    -------
    dict
        Merged configuration dictionary
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=r".\psf_args.yaml",
        help="Path to YAML config file",
    )

    # CLI arguments
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
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--hl", type=int, help="Half profile length")
    parser.add_argument("--ds", type=int, help="Derivative step size")
    parser.add_argument("--axis_shifts", type=int, help="Number of axis shift steps")
    parser.add_argument("--filter", type=str, help="Reconstruction filter name")
    parser.add_argument("--avg_number", type=int, help="Number of profiles to average")
    parser.add_argument(
        "--sym", action="store_true", default=None, help="Symmetrize the sinogram"
    )
    parser.add_argument(
        "--dtheta",
        type=float,
        help="Angle of circular sector for oversampling in degrees",
    )
    parser.add_argument(
        "--resample1", type=float, help="First resample factor (fine grid)."
    )
    parser.add_argument(
        "--resample2",
        type=float,
        help="Second resample factor (coarse grid). This will be the final oversampling factor.",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        help=" Standard deviation of the gaussian blur applied between the fine and the coarse resampling.",
    )
    parser.add_argument("--show", action="store_true", default=None, help="Show plots")

    parser.add_argument(
        "--oversample_strategy",
        type=int,
        choices=[1, 2],
        help="Choose oversampling strategy: 1 or 2. Default is 1 when oversampling is enabled.",
    )

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
        default=None,
        help="Enable averaging neighboring profiles",
    )
    avg_group.add_argument(
        "--no_avg",
        dest="avg_neighbors",
        action="store_false",
        default=None,
        help="Disable averaging neighboring profiles",
    )

    oversample_group = parser.add_mutually_exclusive_group()
    oversample_group.add_argument(
        "--oversample",
        dest="oversample",
        action="store_true",
        default=None,
        help="Enable oversampling",
    )
    oversample_group.add_argument(
        "--no_oversample",
        dest="oversample",
        action="store_false",
        default=None,
        help="Disable oversampling",
    )

    args, unknown = parser.parse_known_args()

    # 1. Set code defaults (lowest priority)
    config = {
        "img_path": None,
        "auto_shift": True,
        "manual_shift": None,
        "no_shift": False,
        "avg_neighbors": True,
        "oversample": False,
        "no_hough": False,
        "symmetrize": False,
        "show_plots": False,
        "oversample_strategy": 1,
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
        "nangles": "n_angles",
        "hl": "profile_half_length",
        "ds": "derivative_step",
        "axis_shifts": "axis_shifts",
        "filter": "filter_name",
        "sym": "symmetrize",
        "show": "show_plots",
        "avg_neighbors": "avg_neighbors",
        "avg_number": "avg_number",
        "oversample": "oversample",
        "oversample_strategy": "oversample_strategy",
        "dtheta": "dtheta",
        "resample1": "resample1",
        "resample2": "resample2",
        "gaussian_sigma": "gaussian_sigma",
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

    # Check which CLI flags were *actually passed*
    cli_manual_shift = cli_dict.get("manual_shift")
    cli_auto_shift = cli_dict.get("auto_shift")
    cli_no_shift = cli_dict.get("no_shift")

    if cli_manual_shift is not None:
        # CLI --manual_shift was used
        config["auto_shift"] = False
        config["no_shift"] = False
        config["manual_shift"] = cli_manual_shift
    elif cli_auto_shift is True:
        # CLI --auto_shift was used
        config["auto_shift"] = True
        config["no_shift"] = False
        config["manual_shift"] = None
    elif cli_no_shift is True:
        # CLI --no_shift was used
        config["auto_shift"] = False
        config["no_shift"] = True
        config["manual_shift"] = None
    # If no CLI shift flag was passed, the config (from YAML or default) is used as-is.

    return config


def validate_args(args: dict) -> None:
    """
    Validate the merged configuration arguments.
    Parameters
    ----------
    args
        Merged configuration dictionary

    Raises
    ------
    ValueError
        If any argument is invalid
    """
    if not args.get("img_path"):
        raise ValueError("Image path is required. Use --f to specify the image file.")
    if args.get("pixel_size") is None or args["pixel_size"] <= 0:
        raise ValueError("Pixel size must be a positive number.")
    if args.get("circle_diameter") is None or args["circle_diameter"] <= 0:
        raise ValueError("Circle diameter must be a positive number.")
    if args.get("n_angles") is None or args["n_angles"] <= 0:
        raise ValueError("Number of angles must be a positive integer.")
    if args.get("profile_half_length") is None or args["profile_half_length"] <= 0:
        raise ValueError("Half profile length must be a positive integer.")
    if args.get("derivative_step") is None or args["derivative_step"] <= 0:
        raise ValueError("Derivative step size must be a positive integer.")

    avg_num = args.get("avg_number")
    if args.get("avg_neighbors") and avg_num is not None:
        if avg_num <= 0 or avg_num % 2 == 0:
            raise ValueError("Average number must be a positive odd integer.")

    if args.get("manual_shift") is not None and not isinstance(
        args["manual_shift"], int
    ):
        raise ValueError("Manual shift must be an integer.")

    # Validation for oversampling args
    if args.get("oversample"):
        if args.get("dtheta") is None or args["dtheta"] <= 0:
            raise ValueError("dtheta must be a positive number for oversampling.")
        if args.get("resample1") is None or args["resample1"] <= 0:
            raise ValueError("resample1 must be a positive number for oversampling.")
        if args.get("resample2") is None or args["resample2"] <= 0:
            raise ValueError("resample2 must be a positive number for oversampling.")
        if args.get("gaussian_sigma") is None or args["gaussian_sigma"] < 0:
            raise ValueError("gaussian_sigma must be non-negative for oversampling.")
