import sys
from pathlib import Path

import pytest
import yaml

from scopexr.arg_parser_psf import get_merged_config, validate_args

# Ensure local src is on the path when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    yaml_path = tmp_path / "psf_args.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    return yaml_path


def _valid_config() -> dict:
    return {
        "img_path": "image.raw",
        "out_dir": "./output",
        "pixel_size": 0.1,
        "circle_diameter": 1.0,
        "n_angles": 180,
        "profile_half_length": 50,
        "derivative_step": 1,
        "axis_shifts": 5,
        "filter_name": "hamming",
        "auto_shift": True,
        "manual_shift": None,
        "no_shift": False,
        "no_hough": False,
        "symmetrize": False,
        "show_plots": False,
        "oversample": False,
        "dtheta": 10.0,
        "resample2": 1.5,
        "gaussian_sigma": 0.5,
    }


def test_cli_overrides_yaml_and_defaults(tmp_path, monkeypatch):
    yaml_data = {
        "img_path": "yaml_image.raw",
        "out_dir": "./yaml_out",
        "pixel_size": 0.2,
        "circle_diameter": 1.1,
        "auto_shift": False,
        "manual_shift": 3,
        "n_angles": 200,
        "oversample": True,
        "dtheta": 15.0,
    }
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = [
        "prog",
        "--config",
        str(yaml_path),
        "--f",
        "cli_image.raw",
        "--o",
        "./cli_out",
        "--p",
        "0.4",
        "--auto_shift",
        "--no_oversample",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["img_path"] == "cli_image.raw"
    assert config["out_dir"] == "./cli_out"
    assert config["pixel_size"] == 0.4  # CLI overrides YAML
    assert config["circle_diameter"] == 1.1  # YAML retained
    assert config["auto_shift"] is True  # CLI flag wins
    assert config["manual_shift"] is None  # auto_shift clears manual
    assert config["n_angles"] == 200  # YAML retained
    assert config["oversample"] is False  # CLI flag overrides YAML
    assert config["dtheta"] == 15.0  # YAML retained


def test_manual_shift_cli_overrides_yaml(tmp_path, monkeypatch):
    yaml_data = {"auto_shift": True, "manual_shift": None, "no_shift": False}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = ["prog", "--config", str(yaml_path), "--manual_shift", "7"]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["manual_shift"] == 7
    assert config["auto_shift"] is False
    assert config["no_shift"] is False


def test_no_shift_cli_flag(tmp_path, monkeypatch):
    yaml_data = {"auto_shift": True, "manual_shift": None, "no_shift": False}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = ["prog", "--config", str(yaml_path), "--no_shift"]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["no_shift"] is True
    assert config["auto_shift"] is False
    assert config["manual_shift"] is None


def test_oversample_cli_flag(tmp_path, monkeypatch):
    yaml_data = {
        "oversample": False,
        "dtheta": 10.0,
        "resample2": 1.5,
        "gaussian_sigma": 0.5,
    }
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = [
        "prog",
        "--config",
        str(yaml_path),
        "--oversample",
        "--dtheta",
        "20.0",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["oversample"] is True
    assert config["dtheta"] == 20.0  # CLI overrides YAML


def test_validate_args_success():
    cfg = _valid_config()
    validate_args(cfg)  # Should not raise


def test_validate_args_with_oversample():
    cfg = _valid_config()
    cfg["oversample"] = True
    validate_args(cfg)  # Should not raise


@pytest.mark.parametrize(
    "mutations,expected_message",
    [
        ({"img_path": None}, "Image path is required"),
        ({"pixel_size": 0}, "Pixel size must be a positive number"),
        ({"circle_diameter": 0}, "Circle diameter must be a positive number"),
        ({"n_angles": 0}, "Number of angles must be a positive integer"),
        ({"profile_half_length": 0}, "Half profile length must be a positive integer"),
        ({"derivative_step": 0}, "Derivative step size must be a positive integer"),
        ({"manual_shift": 1.5}, "Manual shift must be an integer"),
        (
            {"oversample": True, "dtheta": 0},
            "dtheta must be a positive number for oversampling",
        ),
        (
            {"oversample": True, "dtheta": 10.0, "resample2": 0},
            "resample2 must be a positive number for oversampling",
        ),
        (
            {
                "oversample": True,
                "dtheta": 10.0,
                "resample2": 1.5,
                "gaussian_sigma": -1,
            },
            "gaussian_sigma must be non-negative for oversampling",
        ),
    ],
)
def test_validate_args_failures(mutations, expected_message):
    cfg = _valid_config()
    cfg.update(mutations)
    with pytest.raises(ValueError, match=expected_message):
        validate_args(cfg)


def test_symmetrize_and_show_plots_flags(tmp_path, monkeypatch):
    yaml_data = {"symmetrize": False, "show_plots": False}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = ["prog", "--config", str(yaml_path), "--sym", "--show"]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["symmetrize"] is True
    assert config["show_plots"] is True


def test_no_hough_flag(tmp_path, monkeypatch):
    yaml_data = {"no_hough": False}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = ["prog", "--config", str(yaml_path), "--no_hough"]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["no_hough"] is True


def test_all_numeric_params(tmp_path, monkeypatch):
    yaml_data = {}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = [
        "prog",
        "--config",
        str(yaml_path),
        "--nangles",
        "360",
        "--hl",
        "100",
        "--ds",
        "2",
        "--axis_shifts",
        "10",
        "--dtheta",
        "25.5",
        "--resample2",
        "2.5",
        "--gaussian_sigma",
        "1.2",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["n_angles"] == 360
    assert config["profile_half_length"] == 100
    assert config["derivative_step"] == 2
    assert config["axis_shifts"] == 10
    assert config["dtheta"] == 25.5
    assert config["resample2"] == 2.5
    assert config["gaussian_sigma"] == 1.2


def test_filter_name_cli(tmp_path, monkeypatch):
    yaml_data = {"filter_name": "hamming"}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = ["prog", "--config", str(yaml_path), "--filter", "hann"]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["filter_name"] == "hann"


def test_config_file_not_found(tmp_path, monkeypatch, capsys):
    non_existent_path = tmp_path / "non_existent.yaml"

    cli_args = ["prog", "--config", str(non_existent_path)]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    # Should still return a config with defaults
    assert config["auto_shift"] is True
    assert config["no_shift"] is False

    # Check warning was printed to stderr
    captured = capsys.readouterr()
    assert "Warning: Config file not found" in captured.err
    assert str(non_existent_path) in captured.err


def test_invalid_yaml_file(tmp_path, monkeypatch, capsys):
    # Create an invalid YAML file
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text("invalid: yaml: content: [unclosed")

    cli_args = ["prog", "--config", str(yaml_path)]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    # Should still return a config with defaults
    assert config is not None

    # Check error was printed to stderr
    captured = capsys.readouterr()
    assert "Error loading YAML config" in captured.err


def test_gaussian_sigma_none_allowed():
    """Test that gaussian_sigma can be None (treated as 0)."""
    cfg = _valid_config()
    cfg["oversample"] = True
    cfg["gaussian_sigma"] = None
    validate_args(cfg)  # Should not raise


def test_gaussian_sigma_zero_allowed():
    """Test that gaussian_sigma can be 0 (no blur)."""
    cfg = _valid_config()
    cfg["oversample"] = True
    cfg["gaussian_sigma"] = 0.0
    validate_args(cfg)  # Should not raise
