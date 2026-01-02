import sys
from pathlib import Path

import pytest
import yaml

# Ensure local src is on the path when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scopexr.arg_parser_fs import get_merged_config, validate_args


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    yaml_path = tmp_path / "fs_args.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    return yaml_path


def _valid_config() -> dict:
    return {
        "img_path": "image.raw",
        "out_dir": "./output",
        "pixel_size": 0.1,
        "circle_diameter": 1.0,
        "magnification": 1.0,
        "min_n": 10,
        "n_angles": 180,
        "profile_half_length": 50,
        "derivative_step": 1,
        "axis_shifts": 5,
        "filter_name": "hamming",
        "auto_shift": True,
        "manual_shift": None,
        "no_shift": False,
        "avg_neighbors": False,
        "avg_number": 3,
        "no_hough": False,
        "symmetrize": False,
        "show_plots": False,
        "hough_params": {},
    }


def test_cli_overrides_yaml_and_defaults(tmp_path, monkeypatch):
    yaml_data = {
        "img_path": "yaml_image.raw",
        "out_dir": "./yaml_out",
        "pixel_size": 0.2,
        "circle_diameter": 1.1,
        "magnification": 1.5,
        "avg_neighbors": False,
        "avg_number": 5,
        "auto_shift": False,
        "manual_shift": 3,
        "n_angles": 200,
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
        "--avg",
        "--auto_shift",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["img_path"] == "cli_image.raw"
    assert config["out_dir"] == "./cli_out"
    assert config["pixel_size"] == 0.4  # CLI overrides YAML
    assert config["circle_diameter"] == 1.1  # YAML retained
    assert config["magnification"] == 1.5  # YAML retained
    assert config["avg_neighbors"] is True  # CLI flag
    assert config["avg_number"] == 5  # YAML retained
    assert config["auto_shift"] is True  # CLI flag wins
    assert config["manual_shift"] is None  # auto_shift clears manual
    assert config["n_angles"] == 200  # YAML retained
    assert config["min_n"] == 100  # default retained


def test_manual_shift_cli_overrides_yaml(tmp_path, monkeypatch):
    yaml_data = {"auto_shift": True, "manual_shift": None, "no_shift": False}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = ["prog", "--config", str(yaml_path), "--manual_shift", "7"]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["manual_shift"] == 7
    assert config["auto_shift"] is False
    assert config["no_shift"] is False


def test_validate_args_success():
    cfg = _valid_config()
    validate_args(cfg)  # Should not raise


@pytest.mark.parametrize(
    "mutations,expected_message",
    [
        ({"img_path": None}, "Image path is required"),
        ({"pixel_size": 0}, "Pixel size must be a positive number"),
        ({"circle_diameter": 0}, "Circle diameter must be a positive number"),
        ({"magnification": -1}, "Magnification must be a positive number"),
        ({"min_n": 0}, "Minimum pixel count must be a positive integer"),
        ({"n_angles": 0}, "Number of angles must be a positive integer"),
        ({"profile_half_length": 0}, "Half profile length must be a positive integer"),
        ({"derivative_step": 0}, "Derivative step size must be a positive integer"),
        ({"axis_shifts": -1}, "Axis shifts must be a non-negative integer"),
        (
            {"avg_neighbors": True, "avg_number": 2},
            "Average number must be a positive odd integer",
        ),
        ({"manual_shift": 1.5}, "Manual shift must be an integer"),
    ],
)
def test_validate_args_failures(mutations, expected_message):
    cfg = _valid_config()
    cfg.update(mutations)
    with pytest.raises(ValueError, match=expected_message):
        validate_args(cfg)


def test_no_shift_cli_overrides_yaml(tmp_path, monkeypatch):
    yaml_data = {"auto_shift": True, "manual_shift": None, "no_shift": False}
    yaml_path = _write_yaml(tmp_path, yaml_data)

    cli_args = ["prog", "--config", str(yaml_path), "--no_shift"]
    monkeypatch.setattr(sys, "argv", cli_args)

    config = get_merged_config()

    assert config["no_shift"] is True
    assert config["auto_shift"] is False
    assert config["manual_shift"] is None


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
