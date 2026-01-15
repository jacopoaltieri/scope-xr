# Changelog

All notable changes to SCOPE-XR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.6] - 2026-01-15

### Changed

- Various speed-ups using vectorized operations
- Improved angle selection for FS recon
- Improved code comments for readability (only for devs)
- Improved file linting (only for devs)

## [1.1.5] - 2026-01-09

### Added

- Comprehensive documentation improvements
- Quickstart guide for new users
- Troubleshooting section
- References and citations documentation
- Contributing guidelines
- This changelog file
- "Advanced" tab in the GUI to modify Hough circle detection parameters
  

### Changed

- Developer dependencies now include ruff for code linting
- Version bump for documentation updates
- Improved project metadata

## [1.1.0] - 2025

### Added

- GUI interface with PyQt6
- Focal spot analysis module
- PSF (Point Spread Function) analysis module
- Support for multiple image formats (.png, .tif, .raw)
- Automatic circle detection using Hough Transform
- Filtered back-projection reconstruction
- Configurable YAML-based parameter system
- CLI tools: `scopexr-fs`, `scopexr-psf`, `scopexr-gui`
- Comprehensive test suite
- Sphinx documentation

### Features

#### Focal Spot Analysis

- 2D focal spot reconstruction
- Edge-derivative method implementation
- Multiple reconstruction filters (ramp, Shepp-Logan, Hamming, etc.)
- Automatic magnification estimation
- Sinogram symmetrization
- Axis shift correction for centering errors
- Profile averaging for noise reduction
- FWHM measurements in horizontal and vertical directions

#### PSF Analysis

- 2D detector PSF reconstruction
- Sub-pixel oversampling capability
- MTF (Modulation Transfer Function) calculation
- Gaussian fitting for PSF characterization
- Azimuthal profile analysis

#### GUI Features

- Separate tabs for Focal Spot and PSF analysis
- Image preview functionality
- Interactive parameter editing
- Live console output
- Direct YAML config file editing
- Batch-friendly design

#### CLI Features

- Full parameter override capability
- YAML configuration file support
- Batch processing ready
- Comprehensive help documentation

### Technical Improvements

- Robust error handling
- Progress reporting
- Automatic output directory creation
- Results saved as PNG and TIFF images and text files

## [1.0.0] - 2024 (Initial Release)

### Added

- Basic focal spot and PSF analysis functionality
- Initial implementation of reconstruction algorithms
- Command-line interface
- Core image processing utilities

---

## Version Numbering

SCOPE-XR follows Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

## Types of Changes

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

## Links

- [Repository](https://github.com/jacopoaltieri/scope-xr)
- [Documentation](https://scope-xr.readthedocs.io/)
- [Issue Tracker](https://github.com/jacopoaltieri/scope-xr/issues)

## Reporting Issues

If you encounter bugs or have feature requests, please open an issue on GitHub with:

- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- SCOPE-XR version and Python version
- Operating system

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to SCOPE-XR.
