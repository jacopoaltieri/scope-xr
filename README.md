# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
  - [Installing spot-xr](#installing-spot-xr)
- [Usage](#usage)
    - [Default Execution](#default-execution)
    - [Supported Image Formats](#supported-image-formats)
    - [Overriding Configuration Parameters](#overriding-configuration-parameters)
  - [Available CLI Flags](#available-cli-flags)
    - [Focal Spot CLI](#focal-spot-cli)
    - [PSF CLI](#psf-cli)
  - [Processing Pipeline](#processing-pipeline)
    - [1. Input Image](#1-input-image)
    - [2. Circle Detection Check](#2-circle-detection-check)
    - [3. Sinogram and Profiles](#3-sinogram-and-profiles)
    - [4. Reconstruction via Filtered Back Projection (FBP)](#4-reconstruction-via-filtered-back-projection-fbp)
    - [5.1 Focal Spot Dimension Measurement](#51-focal-spot-dimension-measurement)
    - [5.2 PSF measurements](#52-psf-measurements)

---

# Introduction 
SPOT-XR (Single-image Profiling Of Tube X-ray sources and detector Response) is a Python package created to compute the Focal Spot dimensions of a X-ray tube or the PSF response of a detector starting from a single acquisition of a circular cut-out or disk test object. This package aims to:
- Focal Spot: automate the image analysis process first developed by [Di Domenico et al.](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.4938414) and available in the form of an [ImageJ plugin](https://medical-physics.unife.it/downloads/imagej-plugins)
- PSF: provide the code for the method proposed by [Forster et al.](https://www.researchgate.net/publication/387092230_Single-shot_2D_detector_point-spread_function_analysis_employing_a_circular_aperture_and_a_back-projection_approach)

## Installing spot-xr

Since this package is not distributed yet, users will need to clone the GitHub repository and install the required packages before using it.

Run the command:

```bash
git clone "https://github.com/jacopoaltieri/spot-xr"
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

# Usage

The program supports configurable execution via a YAML configuration file. For each method (**Focal Spot** and **PSF**), a different YAML file is used, each containing its own set of parameters. By default, the program loads the respective file and uses the defined parameters. Additionally, command-line arguments (CLI) can be used to override specific settings at runtime. This allows for setup-specific configuration without the need to repeatedly pass the same arguments.

### Default Execution

To run the program with the default settings (as defined in `fs_args.yaml` or `psf_args.yaml`), use the following commands:

- **Focal Spot:**
  ```bash
  python fs_main.py --f "path/to/img.png"
  ```

- **PSF:**
  ```bash
  python psf_main.py --f "path/to/img.png"
  ```

### Supported Image Formats

The supported input image formats are:

- `.png`
- `.tif`
- `.raw` (must be accompanied by a corresponding `.xml` metadata file)

### Overriding Configuration Parameters

You can override any configuration value directly from the command line by adding the corresponding flag. For example:

```bash
python fs_main.py --f "path/to/img.png" --p 0.2
```

In this case, the pixel size will be set to `0.2 mm` instead of the default value specified in the YAML file.


## Available CLI Flags
### Focal Spot CLI

| **Flag**                | **Description**                                                                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--config` (str)        | Path to the YAML configuration file.                                                                                                          |
| `--f` (str, *required*) | Path to the input image file (`.raw`, `.png`, `.tif`).                                                                                        |
| `--o` (str)             | Output directory to store results.                                                                                                            |
| `--p` (float)           | Pixel size in mm.                                                                                                                             |
| `--d` (float)           | Physical diameter of the circular object in mm.                                                                                               |
| `--no_hough`            | Skip Hough Transform for automatic circle detection.                                                                                          |
| `--m` (float)           | Image magnification. If not provided, estimated automatically. Providing it from geometrical considerations may lead to more precise results. |
| `--n` (int)             | Minimum number of pixels required to achieve a reasonable focal spot size.                                                                    |
| `--nangles` (int)       | Number of angular projections for profile extraction.                                                                                         |
| `--hl` (int)            | Half length of the extracted radial profiles.                                                                                                 |
| `--ds` (int)            | Step size used for numerical derivative calculations.                                                                                         |
| `--axis_shifts` (int)   | Number of steps to shift the sinogram axis.                                                                                                   |
| `--filter` (str)        | Filter used during focal spot reconstruction. Options: `ramp`, `shepp-logan`, `cosine`, `hamming`, `hann`. Use `None` for no filter.          |
| `--sym`                 | Symmetrize the sinogram before reconstruction.                                                                                                |
| `--shift`               | Enable automatic sinogram shifting.                                                                                                           |
| `--no_shift`            | Disable automatic sinogram shifting. (*Mutually exclusive with* `--shift`)                                                                    |
| `--avg`                 | Average neighboring sinogram profiles to improve FWHM estimation.                                                                             |
| `--no_avg`              | Do not average neighboring profiles. (*Mutually exclusive with* `--avg`)                                                                      |
| `--show`                | Display plots during processing (matplotlib windows).                                                                                         |

### PSF CLI

| **Flag**                | **Description**                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `--config` (str)        | Path to the YAML configuration file.                                                                                                 |
| `--f` (str, *required*) | Path to the input image file (`.raw`, `.png`, `.tif`).                                                                               |
| `--o` (str)             | Output directory to store results.                                                                                                   |
| `--p` (float)           | Pixel size in mm.                                                                                                                    |
| `--d` (float)           | Physical diameter of the circular object in mm.                                                                                      |
| `--no_hough`            | Skip Hough Transform for automatic circle detection.                                                                                 |
| `--nangles` (int)       | Number of angular projections for profile extraction.                                                                                |
| `--hl` (int)            | Half length of the extracted radial profiles.                                                                                        |
| `--ds` (int)            | Step size used for numerical derivative calculations.                                                                                |
| `--axis_shifts` (int)   | Number of steps to shift the sinogram axis.                                                                                          |
| `--filter` (str)        | Filter used during focal spot reconstruction. Options: `ramp`, `shepp-logan`, `cosine`, `hamming`, `hann`. Use `None` for no filter. |
| `--sym`                 | Symmetrize the sinogram before reconstruction.                                                                                       |
| `--dtheta`              | Angle of the circular sector for oversampling (in degrees).                                                                          |
| `--resample1`           | First resample factor (fine grid).                                                                                                   |
| `--resample2`           | Second resample factor (coarse grid). This will be the final oversampling factor.                                                    |
| `--gaussian_sigma`      | Standard deviation of the gaussian blur applied between the fine and the coarse resampling.                                          |
| `--shift`               | Enable automatic sinogram shifting.                                                                                                  |
| `--no_shift`            | Disable automatic sinogram shifting. (*Mutually exclusive with* `--shift`)                                                           |
| `--avg`                 | Average neighboring sinogram profiles to improve FWHM estimation.                                                                    |
| `--no_avg`              | Do not average neighboring profiles. (*Mutually exclusive with* `--avg`)                                                             |
| `--oversample`          | Performs oversampling.                                                                                                               |
| `--no_oversample`       | Disables oversampling. (*Mutually exclusive with* `--oversample`)                                                                                                                 |
| `--show`                | Display plots during processing (matplotlib windows).                                                                                |

## Processing Pipeline

### 1. Input Image

The input image can be a `.raw`, `.png` or `.tif` grayscale image of the test object. The package automatically identifies the circular aperture via a Hough Transform; then a square region is cropped around the detected circle and the center and radius of the aperture are computed through a center-of-mass estimation. If the automatic detection fails, or if the user wants to pass the already cropped image, one can simply use the flag `--no_hough`. The parameters of the Hough transform have been empirically chosen to work well with these kinds of images. The user is free to change them by editing the corresponding section in the `args.yaml` file.

### 2. Circle Detection Check

Once the center and radius of the circle are computed, the program checks if the estimated radius satisfies the straight-edge constraint (for focal spot reconstruction only). Then, it extracts the radial profiles and their derivative.  
If the circular edge is not perfectly centered, the sinogram could be shifted and the focal spot reconstruction may not work properly. For this reason, the script exploits the symmetry of the sinogram to compute the best axis shift.

### 3. Sinogram and Profiles

- **Radial Profiles Extraction:**  
  Radial profiles are extracted from the cropped region.

- **Derivative and Sinogram Generation:**  
  The derivative of each radial profile is computed, and these derivatives are assembled into a sinogram.

### 4. Reconstruction via Filtered Back Projection (FBP)

The reconstruction is obtained via a Filtered Back Projection algorithm on the best-shifted sinogram with different selectable filters. The program also produces a sequence of reconstructions with different shifts to be checked manually.

### 5.1 Focal Spot Dimension Measurement

To compute the two dimensions of the focal spot, the program fits an ERF function on each radial profile and computes the ones with the largest slope value.; the narrow profile is taken perpendicular to the wide one. Then it identifies these profiles on the sinogram and the reconstruction based on their angle index.

- **Sinogram FWHM Measurement:**  
  Direct measurement of the Full Width at Half Maximum (FWHM) of the sinogram profile. The profile can be averaged with its nearest neighbors to reduce noise. This method may be imprecise or underestimate the actual size of the focal spot due to the low number of sample points.

- **ERF-Based FWHM Measurement:**  
  Computation of the FWHM from the σ parameter of the ERF fit. Due to the higher quality of the edge profile signal (compared to the sinogram), this method is considered to give more reliable results.


From the FWHM measurement, the focal spot dimensions are evaluated as:

```
fs = (FWHM * P) / M_fs
```

Where:  
- `P` is the pixel size (mm).  
- `M_fs` is the magnification of the focal spot on the image plane, computed as:

```
M_fs = M - 1
```

If the magnification `M` is not passed as a parameter by the user, the program computes it directly from the test object radius and the estimated radius on the image.


### 5.2 PSF measurements
For the PSF measurements, the program finds the horizontal and vertical profiles and computes their FWHM by fitting a gaussian curve on the corresponding sinogram profiles.

if the *--oversample* flag is checked, the program performs subpixel resolution sampling while computing the radial profiles.

---
