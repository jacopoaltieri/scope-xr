# Contributing to SCOPE-XR

Thank you for your interest in contributing to SCOPE-XR! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guide](#code-style-guide)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Expected Behavior

- Be respectful and considerate
- Use welcoming and inclusive language
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal or political attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Familiarity with X-ray imaging concepts (helpful but not required)

### Setting Up Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork:**

   ```bash
   git clone https://github.com/YOUR-USERNAME/scope-xr.git
   cd scope-xr
   ```

3. **Add upstream remote:**

   ```bash
   git remote add upstream https://github.com/jacopoaltieri/scope-xr.git
   ```

4. **Create a virtual environment:**

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install in development mode with all dependencies:**

   ```bash
   pip install -e .[dev,test,docs]
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

**Branch naming conventions:**

- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### Making Changes

1. Make your changes in logical, atomic commits
2. Write clear, descriptive commit messages
3. Test your changes thoroughly
4. Update documentation as needed

### Keeping Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Code Style Guide

SCOPE-XR follows Python community standards and uses **ruff** for linting.

### Running Code Quality Checks

Before committing, run:

```bash
# Check for linting issues
ruff check

# Auto-fix issues where possible
ruff check --fix

# Format code (if using ruff formatter)
ruff format
```

### Style Guidelines

**General Principles:**

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Keep functions focused and concise
- Add docstrings to all public functions and classes

**Naming Conventions:**

```python
# Variables and functions: snake_case
pixel_size = 0.2
def calculate_fwhm(profile):
    pass

# Classes: PascalCase
class FocalSpotAnalyzer:
    pass

# Constants: UPPER_CASE
MAX_ITERATIONS = 100
DEFAULT_FILTER = "ramp"

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

**Docstring Format:**

Use NumPy-style docstrings:

```python
def reconstruct_focal_spot(sinogram, filter_type="ramp"):
    """
    Reconstruct focal spot using filtered back-projection.

    Parameters
    ----------
    sinogram : numpy.ndarray
        2D array of shape (n_angles, n_detectors) containing sinogram data.
    filter_type : str, optional
        Reconstruction filter to use. Default is "ramp".
        Options: "ramp", "shepp-logan", "hamming", "hann".

    Returns
    -------
    reconstruction : numpy.ndarray
        2D reconstructed focal spot image.

    Raises
    ------
    ValueError
        If sinogram has invalid shape or filter_type is not recognized.

    Examples
    --------
    >>> sino = np.random.rand(360, 500)
    >>> fs = reconstruct_focal_spot(sino, filter_type="hamming")
    >>> print(fs.shape)
    (500, 500)
    """
    pass
```

**Import Organization:**

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Local imports
from scopexr.utils import load_image
from scopexr.circle_detection import detect_circle
```

## Testing

### Running Tests

SCOPE-XR uses **pytest** for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scopexr --cov-report=html

# Run specific test file
pytest tests/test_circle_detection.py

# Run specific test
pytest tests/test_utils.py::test_load_image

# Run with verbose output
pytest -v
```

### Writing Tests

Create test files in the `tests/` directory with the prefix `test_`:

```python
# tests/test_my_feature.py
import pytest
import numpy as np
from scopexr.my_module import my_function


def test_my_function_basic():
    """Test basic functionality."""
    result = my_function(input_data)
    assert result == expected_output


def test_my_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        my_function(invalid_input)


@pytest.mark.parametrize("input_val,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_my_function_parametrized(input_val, expected):
    """Test with multiple inputs."""
    assert my_function(input_val) == expected
```

**Test Guidelines:**

- Write tests for all new functionality
- Test edge cases and error conditions
- Use descriptive test names
- Keep tests independent and isolated
- Mock external dependencies when appropriate

## Documentation

### Building Documentation

Documentation is built using Sphinx:

```bash
cd docs
make html
# Open docs/build/html/index.html in browser
```

### Documentation Standards

- Update docstrings for any changed functions
- Add examples to docstrings when helpful
- Update relevant `.rst` files in `docs/source/`
- Add new modules to API documentation
- Include references for scientific methods

### Adding New Documentation Pages

1. Create `.rst` file in `docs/source/`
2. Add to appropriate `toctree` in `index.rst`
3. Build and verify locally
4. Include in pull request

## Submitting Changes

### Before Submitting

**Checklist:**

- [ ] Code follows style guidelines (ruff passes)
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] Changes are on a feature branch
- [ ] Branch is up-to-date with upstream main

### Creating a Pull Request

1. **Push your branch to your fork:**

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request on GitHub**

3. **Fill out the PR template** with:
   - Clear description of changes
   - Related issue numbers (if applicable)
   - Any breaking changes
   - Screenshots (for GUI changes)

4. **Respond to review feedback**

### Pull Request Guidelines

**Title Format:**

```
[Type] Brief description

Examples:
[Feature] Add sub-pixel oversampling for PSF analysis
[Bugfix] Fix circle detection for low-contrast images
[Docs] Add troubleshooting guide
```

**Description Template:**

```markdown
## Description
Brief summary of changes

## Motivation
Why is this change needed?

## Changes
- List of specific changes
- Another change

## Testing
How were these changes tested?

## Related Issues
Closes #123
Related to #456

## Screenshots (if applicable)
[Add screenshots for GUI changes]

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
```

## Reporting Bugs

### Before Reporting

1. Check if the issue already exists
2. Verify it's not a configuration problem
3. Test with the latest version
4. Gather relevant information

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. Use configuration '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
- SCOPE-XR version: [e.g., 1.1.5]
- Python version: [e.g., 3.10.5]
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Installation method: [pip, source]

**Configuration file (if relevant):**
 Paste relevant YAML config


**Error message:**
Paste full error traceback


**Additional context**
Any other relevant information.
```

## Feature Requests

We welcome feature suggestions! Please provide:

- Clear description of the feature
- Use case / motivation
- Expected behavior
- Potential implementation approach (optional)
- Relevant scientific references (if applicable)

## Questions and Support

- **Documentation:** Check [https://scope-xr.readthedocs.io](https://scope-xr.readthedocs.io)
- **GitHub Issues:** For bug reports and feature requests
- **Discussions:** For questions and general discussion

## Recognition

Contributors will be acknowledged in:

- Release notes
- Documentation contributors page

## License

By contributing to SCOPE-XR, you agree that your contributions will be licensed under the GNU General Public License v3.0.

---

**Thank you for contributing to SCOPE-XR!**

Your contributions help advance X-ray imaging research and make quality assurance more accessible to the community.
