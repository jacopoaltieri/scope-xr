import argparse
import sys
import yaml
import os

def get_merged_config():
    parser = argparse.ArgumentParser(description="PSF Analysis Tool")

    # 1. Config Arguments
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to YAML config file. Defaults to local 'psf_args.yaml' or internal package default."
    )
    
    parser.add_argument(
        "--init_config", 
        action="store_true", 
        help="Generate a default 'psf_args.yaml' file in the current directory and exit."
    )

    # 2. CLI Arguments
    parser.add_argument("--f", type=str, help="Path to the image file (.raw/.png/.tif)")
    parser.add_argument("--o", type=str, help="Output directory")
    parser.add_argument("--p", type=float, help="Pixel size in mm")
    parser.add_argument("--d", type=float, help="Circle diameter in mm")
    parser.add_argument("--no_hough", action="store_true", default=None, help="Skip Hough transform detection")
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--hl", type=int, help="Half profile length")
    parser.add_argument("--ds", type=int, help="Derivative step size")
    parser.add_argument("--axis_shifts", type=int, help="Number of axis shift steps")
    parser.add_argument("--filter", type=str, help="Reconstruction filter name")
    parser.add_argument("--avg_number", type=int, help="Number of profiles to average")
    parser.add_argument("--sym", action="store_true", default=None, help="Symmetrize the sinogram")
    parser.add_argument("--show", action="store_true", default=None, help="Show plots")

    # Oversampling parameters
    parser.add_argument("--dtheta", type=float, help="Angle of circular sector for oversampling in degrees")
    parser.add_argument("--resample1", type=float, help="First resample factor (fine grid).")
    parser.add_argument("--resample2", type=float, help="Second resample factor (coarse grid). Final oversampling factor.")
    parser.add_argument("--gaussian_sigma", type=float, help="Standard deviation of gaussian blur between resamples.")
    parser.add_argument("--oversample_strategy", type=int, choices=[1, 2], help="Oversampling strategy: 1 or 2.")

    # Shift Group
    shift_group = parser.add_mutually_exclusive_group()
    shift_group.add_argument("--auto_shift", action="store_true", default=None, help="Enable automatic sinogram centering.")
    shift_group.add_argument("--manual_shift", type=int, default=None, help="Provide a specific manual shift value (in pixels).")
    shift_group.add_argument("--no_shift", action="store_true", default=None, help="Disable all sinogram shifting.")

    # Avg Group
    avg_group = parser.add_mutually_exclusive_group()
    avg_group.add_argument("--avg", dest="avg_neighbors", action="store_true", default=None, help="Enable averaging neighboring profiles")
    avg_group.add_argument("--no_avg", dest="avg_neighbors", action="store_false", default=None, help="Disable averaging neighboring profiles")

    # Oversample Group
    oversample_group = parser.add_mutually_exclusive_group()
    oversample_group.add_argument("--oversample", dest="oversample", action="store_true", default=None, help="Enable oversampling")
    oversample_group.add_argument("--no_oversample", dest="oversample", action="store_false", default=None, help="Disable oversampling")

    args, unknown = parser.parse_known_args()

    # --- SPECIAL CASE: Init Config ---
    if args.init_config:
        return {"init_config": True}

    # --- CONFIG LOADING STRATEGY ---
    # Priority 1: User specified path
    if args.config:
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"[Error] Specified config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
            
    # Priority 2: Local file in current directory
    elif os.path.exists("psf_args.yaml"):
        config_path = "psf_args.yaml"
        print(f"[Info] Using local configuration: {config_path}")
        
    # Priority 3: Internal Package Default
    else:
        config_path = os.path.join(os.path.dirname(__file__), "psf_args.yaml")
        print(f"[Info] Using internal default configuration.")

    # --- LOAD YAML ---
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

    try:
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
    except Exception as e:
        print(f"[Warning] Error loading YAML config from {config_path}: {e}", file=sys.stderr)
        print("Using hardcoded defaults.", file=sys.stderr)

    # --- APPLY CLI OVERRIDES ---
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
        if cli_value is not None:
            config[config_key] = cli_value

    # --- RESOLVE LOGIC CONFLICTS ---
    if cli_dict.get("manual_shift") is not None:
        config["auto_shift"] = False
        config["no_shift"] = False
        config["manual_shift"] = cli_dict.get("manual_shift")
    elif cli_dict.get("auto_shift") is True:
        config["auto_shift"] = True
        config["no_shift"] = False
        config["manual_shift"] = None
    elif cli_dict.get("no_shift") is True:
        config["auto_shift"] = False
        config["no_shift"] = True
        config["manual_shift"] = None

    return config


def validate_args(config):
    # Skip validation if initializing config
    if config.get("init_config"):
        return

    if not config.get("img_path"):
        raise ValueError("Image path is required. Use --f to specify the image file or set 'img_path' in YAML.")
        
    # Basic positive number checks
    positive_checks = [
        ("pixel_size", float),
        ("circle_diameter", float),
        ("n_angles", int),
        ("profile_half_length", int),
        ("derivative_step", int)
    ]

    for key, expected_type in positive_checks:
        val = config.get(key)
        if val is None or val <= 0:
            raise ValueError(f"'{key}' must be a positive {expected_type.__name__}.")

    if config.get("axis_shifts") is not None and config["axis_shifts"] < 0:
        raise ValueError("Axis shifts must be a non-negative integer.")

    avg_num = config.get("avg_number")
    if config.get("avg_neighbors") and avg_num is not None:
        if avg_num <= 0 or avg_num % 2 == 0:
            raise ValueError("Average number must be a positive odd integer.")

    if config.get("manual_shift") is not None and not isinstance(config["manual_shift"], int):
        raise ValueError("Manual shift must be an integer.")

    # Validation for oversampling args
    if config.get("oversample"):
        os_checks = ["dtheta", "resample1", "resample2"]
        for key in os_checks:
            val = config.get(key)
            if val is None or val <= 0:
                raise ValueError(f"'{key}' must be a positive number for oversampling.")
        
        sigma = config.get("gaussian_sigma")
        if sigma is None or sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative for oversampling.")