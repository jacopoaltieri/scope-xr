<h1 align="center">
<img src="https://github.com/jacopoaltieri/scope-xr/blob/main/src/scopexr/scopexr_logo.png?raw=true)" width="300">
</h1><br>

[![PyPI](https://img.shields.io/pypi/v/scopexr)](https://pypi.org/project/scopexr/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Documentation Status](https://img.shields.io/readthedocs/scope-xr)](https://scope-xr.readthedocs.io/en/latest/)

📚 **Full documentation:** [scope-xr.readthedocs.io](https://scope-xr.readthedocs.io/)

- 🚀 [Quickstart Guide](https://scope-xr.readthedocs.io/en/latest/quickstart.html)
- 📖 [User Guide](https://scope-xr.readthedocs.io/en/latest/usage.html)
- 🔧 [Troubleshooting](https://scope-xr.readthedocs.io/en/latest/troubleshooting.html)
- 📝 [API Reference](https://scope-xr.readthedocs.io/en/latest/modules.html)


# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Create and activate a virtual environment (Optional but Recommended):](#create-and-activate-a-virtual-environment-optional-but-recommended)
    - [1. Quick Install (For regular users)](#1-quick-install-for-regular-users)
    - [2. Manual Install (From source)](#2-manual-install-from-source)
- [Usage](#usage)
  - [Supported Image Formats](#supported-image-formats)
  - [GUI Execution](#gui-execution)
  - [CLI Execution](#cli-execution)
    - [Overriding Configuration Parameters](#overriding-configuration-parameters)
  - [Exporting Default Configurations](#exporting-default-configurations)
- [Processing Pipeline](#processing-pipeline)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
    - [BibTeX](#bibtex)
    - [Methodology Citations](#methodology-citations)
- [Changelog](#changelog)
- [⚖️ License](#️-license)

---

# Introduction

**SCOPE-XR (Single-image Characterization Of PErformance in X-Ray systems)** is a specialized Python framework for the automated characterization of X-ray systems. By analyzing a single acquisition of a circular aperture or disk test object, the software reconstructs 2D source distributions and detector responses.

**Key capabilities:**

- **Focal Spot:** Automated reconstruction of 2D focal spot distribution and dimensions based on the methodology by [Di Domenico et al.](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.4938414), which is available in the form of an [ImageJ plugin](https://medical-physics.unife.it/downloads/imagej-plugins).
- **PSF:** Automated reconstruction of 2D PSF distribution of the detector, based on the methodology by [Forster et al.](https://www.researchgate.net/publication/387092230_Single-shot_2D_detector_point-spread_function_analysis_employing_a_circular_aperture_and_a_back-projection_approach), with optional sub-pixel oversampling controlled by a single `gaussian_sigma` parameter (0/None = direct binning, >0 = fine-grid blur).

# Quick Start

Get started in 3 steps:

```bash
# 1. Install from PyPI
pip install scopexr

# 2. Run GUI
scopexr-gui

# 3. Or use CLI
scopexr-fs --f "path/to/image.tif"
```

📘 **New to SCOPE-XR?** Check out the [Quickstart Guide](https://scope-xr.readthedocs.io/en/latest/quickstart.html) with step-by-step instructions.

# Installation

SCOPE-XR is installed as a standard Python package directly from PyPi. It is recommended to use a virtual environment (venv or conda) to keep your system clean.

### Prerequisites

- Python 3.9 or higher
- Git

### Create and activate a virtual environment (Optional but Recommended):

  **Windows:**

  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

  **Linux/macOS:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
### 1. Quick Install (For regular users)

If you just want to use the software without modifying the code, install directly from PyPI:

```bash
pip install scopexr
```

⚠️ **Important**: Configuration files (`.yaml`) are required for execution. Export the bundled defaults to your working directory with:

```bash
scopexr-confex
```

Use `scopexr-confex --output-dir path/to/configs` to export them to a different directory.

### 2. Manual Install (From source)

Recommended if you wish to modify the code or contribute:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/jacopoaltieri/scope-xr
    cd scope-xr
    ```

2. **Install the package:**

    Install in editable mode (recommended for development) to ensure all dependencies are handled automatically:

    ```bash
    pip install -e .
    ```

    If you want also to be able to run the tests or build the docs, run:
    ```bash
    pip install -e .[all]
    ```
# Usage

SCOPE-XR provides two main interfaces designed for different user workflows.

The program supports configurable execution via YAML configuration files.
You can find an example of these files in the `examples` folder, along with some simulated images to test the package.


## Supported Image Formats

The supported input image formats are:

- `.png`
- `.tif` / `.tiff`
- `.raw` (must be accompanied by a corresponding `.xml` metadata file)
- `.dcm` (DICOM)

## GUI Execution

The recommended way for routine analysis. It features live image previews and interactive parameter tuning.
To run the GUI, simply type:

```bash
scopexr-gui
```

GUI Features:

- **Easy Mode Selection**: Separate tabs for "Focal Spot (FS)" and "PSF" analysis.
- **Automatic Configuration**: The GUI automatically loads all default parameters from `fs_args.yaml` or `psf_args.yaml` on startup.
- **Image Preview**: Load .png, .tif/.tiff, or .dcm images to see a preview directly in the app.
- **Full Parameter Control**: All CLI flags are editable via interactive widgets.
- **Edit Config Files**: A button allows you to directly open and edit the default .yaml config file for the active tab.
- **Live Output**: All console output from the analysis script is printed directly to a text box within the GUI.
  
## CLI Execution

Ideal for batch processing and integration into automated research pipelines.
To run the program with the default settings (as defined in `fs_args.yaml` or `psf_args.yaml`), use the following commands:

- **Focal Spot:**

  ```bash
  scopexr-fs --f "path/to/img.png"
  ```

- **PSF:**

  ```bash
  scopexr-psf --f "path/to/img.png"
  ```

Use the `--help` flag with any command to see all available parameters.

### Overriding Configuration Parameters

You can override any configuration value directly from the command line by adding the corresponding flag. For example:

```bash
scopexr-fs --f "path/to/img.png" --p 0.2
```

In this case, the pixel size will be set to `0.2 mm` instead of the default value specified in the YAML file.

The full list of CLI flags is available in the [documentation](https://scope-xr.readthedocs.io/)

## Exporting Default Configurations

The `scopexr-confex` command exports the bundled `fs_args.yaml` and `psf_args.yaml`
configuration files. By default, files are written to the current working directory:

```bash
scopexr-confex
```

To choose another destination, pass `--output-dir` (or `-o`):

```bash
scopexr-confex --output-dir path/to/configs
```


# Processing Pipeline

<p align="center">
  <img src="docs/source/_static/processing_pipeline.png" width="100%" alt="SCOPE-XR Processing Pipeline">
</p>

# Documentation

Comprehensive documentation is available at [scope-xr.readthedocs.io](https://scope-xr.readthedocs.io/):

- **[Quickstart Guide](https://scope-xr.readthedocs.io/en/latest/quickstart.html)** - Step-by-step tutorial for first-time users
- **[Installation](https://scope-xr.readthedocs.io/en/latest/installation.html)** - Detailed installation instructions
- **[Usage Guide](https://scope-xr.readthedocs.io/en/latest/usage.html)** - Complete parameter reference
- **[Troubleshooting](https://scope-xr.readthedocs.io/en/latest/troubleshooting.html)** - Common issues and solutions
- **[Theory](https://scope-xr.readthedocs.io/en/latest/theory.html)** - Physical methodology and equations
- **[Pipeline](https://scope-xr.readthedocs.io/en/latest/pipeline.html)** - Processing workflow details
- **[API Reference](https://scope-xr.readthedocs.io/en/latest/modules.html)** - Module and function documentation


# Contributing

We welcome contributions! Whether you're fixing bugs, adding features, improving documentation, or reporting issues, your help is appreciated.

**Quick Start:**

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/scope-xr.git
cd scope-xr

# Install dev dependencies
pip install -e .[dev,test]

# Create feature branch
git checkout -b feature/my-feature

# Make changes, run tests
ruff check
pytest

# Push and create PR
git push origin feature/my-feature
```

📋 **Full guidelines:** See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions on:
- Development setup
- Code style guide (using ruff)
- Testing requirements
- Pull request process
- Reporting bugs and requesting features

# Citation

If you use **SCOPE-XR** in your research, please cite our published paper as the primary reference:

> Altieri, J., Cardarelli, P., Di Domenico, G., & Taibi, A. (2026). A python framework for single-image characterization of X-ray focal spot distribution and detector point spread function. *Medical Physics*, 53, e70513. https://doi.org/10.1002/mp.70513

### BibTeX

**Primary Paper:**

```bibtex
@article{altieri2026scopexr,
  author  = {Altieri, Jacopo and Cardarelli, Paolo and Di Domenico, Giovanni and Taibi, Angelo},
  title   = {A python framework for single-image characterization of X-ray focal spot distribution and detector point spread function},
  journal = {Medical Physics},
  volume  = {53},
  pages   = {e70513},
  year    = {2026},
  doi     = {10.1002/mp.70513}
}
```

**Software Package (PyPI / Version Specific):**

```bibtex
@software{scopexr2026software,
  author    = {Altieri, Jacopo},
  title     = {SCOPE-XR: Single-image Characterization Of PErformance in X-Ray systems},
  year      = {2026},
  version   = {1.3.6},
  url       = {[https://pypi.org/project/scopexr/](https://pypi.org/project/scopexr/)},
  publisher = {PyPI}
}
```

---

### Methodology Citations

If your work relies specifically on the underlying algorithmic methods, please also consider citing:
- **Focal Spot:** [Di Domenico et al. (2016)](https://doi.org/10.1118/1.4938414)
- **PSF Analysis:** [Forster et al. (2024)](https://doi.org/10.13140/RG.2.2.27408.89607)


# Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

---

# ⚖️ License

Distributed under the GNU General Public License v3.0 (GPL-3.0). See `LICENSE` for the full text.
