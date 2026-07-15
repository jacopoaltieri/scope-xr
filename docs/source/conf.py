from pathlib import Path
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# -- Project information -----------------------------------------------------
project = "scope-xr"
copyright = "2026, Jacopo Altieri"
author = "Jacopo Altieri"
release = "1.3.2"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Core library for generating docs from docstrings
    "sphinx.ext.napoleon",  # Support for Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.githubpages",  # Useful if you host on GitHub Pages
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_use_rtype = False

# -- Options for HTML output -------------------------------------------------
# Using the standard modern theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
