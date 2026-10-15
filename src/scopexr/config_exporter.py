import argparse
from importlib.resources import files
from pathlib import Path
import shutil

CONFIG_FILES = ["fs_args.yaml", "psf_args.yaml"]

def export_configs():
    parser = argparse.ArgumentParser(
        description="Export default scopexr YAML configuration files."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to export config files to (default: current working directory)"
    )

    args = parser.parse_args()
    target_dir: Path = args.output_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in CONFIG_FILES:
        # Navigate inside the default_configs subfolder
        src = files("scopexr").joinpath("default_configs", filename)
        dest = target_dir / filename

        with src.open("rb") as fsrc, open(dest, "wb") as fdest:
            shutil.copyfileobj(fsrc, fdest)

        print(f"Exported {filename} -> {dest}")

if __name__ == "__main__":
    export_configs()