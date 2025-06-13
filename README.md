# Introduction 
SPOT-X (Single-image Profiling Of Tube X-ray sources) is a Python package created to compute the Focal Spot dimensions of a X-ray tube starting from a single acquisition of a circular cut-out or disk test object. This package
aims to automate the image analysis process first developed by [Di Domenico et al.](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.4938414) and available in the form of an [ImageJ plugin](https://medical-physics.unife.it/downloads/imagej-plugins).

## Installing spot-x

Since this package is not distributed yet, users will need to clone the GitHub repository and install the required packages before using it.

Run the command:

```bash
git clone "https://github.com/jacopoaltieri/spot-x"
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

# Usage

The program supports configurable execution via a YAML configuration file. By default, the program loads this file and uses the defined parameters. Additionally, command-line arguments (CLI) can be used to override specific settings at runtime. The YAML file allows for a setup-specific configuration without the need to pass the same arguments each time.

To run the program with the default settings defined in `args.yaml`, execute:

```bash
python main.py --f "path/to/img.png"
```

Supported image formats are `.png`, `.tif` and `.raw` (accompanied by their `.xml` metadata file).

To override any configuration value directly from the command line, simply add the corresponding flag to the command (see [Available CLI Flags](#available-cli-flags) for the list of flags). For example:

```bash
python main.py --f "path/to/img.png" --p 0.2
```

In this case, the value for the pixel size will be set to 0.2 mm rather than the default value.

## Available CLI Flags

| Flag                    | Description                                                                                                                                             |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--config` (str)        | Path to the YAML configuration file.                                                                                                                    |
| `--f` (str, *required*) | Path to the input image file (`.raw`, `.png`, `.tif`).                                                                                                   |
| `--o` (str)             | Output directory to store results.                                                                                                                      |
| `--p` (float)           | Pixel size in mm.                                                                                                                                        |
| `--d` (float)           | Physical diameter of the circular object in mm.                                                                                                         |
| `--no_hough`            | Skip Hough Transform for automatic circle detection.                                                                                                    |
| `--m` (float)           | Image magnification. If not provided, estimated automatically. Providing it from geometrical considerations may lead to more precise results.           |
| `--n` (int)             | Minimum number of pixels required to achieve a reasonable focal spot size.                                                                              |
| `--nangles` (int)       | Number of angular projections for profile extraction.                                                                                                   |
| `--hl` (int)            | Half length of the extracted radial profiles.                                                                                                           |
| `--ds` (int)            | Step size used for numerical derivative calculations.                                                                                                   |
| `--axis_shifts` (int)   | Number of steps to shift the sinogram axis.                                                                                                             |
| `--filter` (str)        | Filter used during focal spot reconstruction. Options: `ramp`, `shepp-logan`, `cosine`, `hamming`, `hann`. Use `None` for no filter.                    |
| `--sym`                 | Symmetrize the sinogram before reconstruction.                                                                                                          |
| `--shift`               | Enable automatic sinogram shifting.                                                                                                                     |
| `--no_shift`            | Disable automatic sinogram shifting. (*Mutually exclusive with `--shift`*)                                                                             |
| `--avg`                 | Average neighboring sinogram profiles to improve FWHM estimation.                                                                                       |
| `--no_avg`              | Do not average neighboring profiles. (*Mutually exclusive with `--avg`*)                                                                               |
| `--show`                | Display plots during processing (matplotlib windows).                                                                                                   |

## Processing Pipeline

### 1. Input Image

The input image can be a `.raw`, `.png` or `.tif` grayscale image of the test object. The package automatically identifies the circular aperture via a Hough Transform; then a square region is cropped around the detected circle and the center and radius of the aperture are computed through a center-of-mass estimation. If the automatic detection fails, or if the user wants to pass the already cropped image, one can simply use the flag `--no_hough`. The parameters of the Hough transform have been empirically chosen to work well with these kinds of images. The user is free to change them by editing the corresponding section in the `args.yaml` file.

### 2. Circle Detection Check

Once the center and radius of the circle are computed, the program checks if the estimated radius satisfies the straight-edge constraint. Then, it extracts the radial profiles and their derivative.  
If the circular edge is not perfectly centered, the sinogram could be shifted and the focal spot reconstruction may not work properly. For this reason, the script exploits the symmetry of the sinogram to compute the best axis shift.

### 3. Sinogram and Profiles

- **Radial Profiles Extraction:**  
  Radial profiles are extracted from the cropped region.

- **Derivative and Sinogram Generation:**  
  The derivative of each radial profile is computed, and these derivatives are assembled into a sinogram.

### 4. Reconstruction via Filtered Back Projection (FBP)

The reconstruction is obtained via a Filtered Back Projection algorithm on the best-shifted sinogram with different selectable filters. The program also produces a sequence of reconstructions with different shifts to be checked manually.

### 5. Focal Spot Dimension Measurement

To compute the two dimensions of the focal spot, the program fits an Error Function (ERF) on each radial profile and finds the profiles with the largest and smallest slope values. It then identifies these profiles on the sinogram and the reconstruction based on their angle index.

- **Sinogram FWHM Measurement:**  
  Direct measurement of the Full Width at Half Maximum (FWHM) of the sinogram profile. The profile can be averaged with its nearest neighbors to reduce noise. This method may be imprecise or underestimate the actual size of the focal spot due to the low number of sample points.

- **ERF-Based FWHM Measurement:**  
  Computation of the FWHM from the σ parameter of the ERF fit. Due to the higher quality of the edge profile signal (compared to the sinogram), this method is considered to give more reliable results.

### 6. Focal Spot Size Calculation

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

---
