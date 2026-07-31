# SCOPE-XR (Single-image Characterization Of PErformance in X-Ray systems)
# Copyright (C) 2026  Jacopo Altieri
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import pydicom


def load_raw_as_ndarray(img_path: str) -> np.ndarray:
    """
    Load a RAW image as a numpy ndarray using metadata from the corresponding XML file.

    Parameters
    ----------
    img_path
        Path to the raw image file (.raw).

    Returns
    -------
    np.ndarray
        2D numpy ndarray representing the image.

    Raises
    ------
    FileNotFoundError
        If the XML metadata file is not found.
    ValueError
        If the XML metadata does not contain valid image dimensions.
    """
    img_path_obj = Path(img_path)
    xml_path = img_path_obj.with_suffix(".xml")

    if not xml_path.exists():
        raise FileNotFoundError(f"Metadata XML not found: '{xml_path}'")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame = root.find("frame")

    if frame is None:
        raise ValueError(f"Missing <frame> element in '{xml_path}'")

    width_element = frame.find("imgWidth")
    height_element = frame.find("imgHeight")

    if width_element is None or width_element.text is None:
        raise ValueError(f"Missing or empty <imgWidth> in '{xml_path}'")

    if height_element is None or height_element.text is None:
        raise ValueError(f"Missing or empty <imgHeight> in '{xml_path}'")

    img_width = int(width_element.text)
    img_height = int(height_element.text)

    with open(img_path, "rb") as f:
        img = np.fromfile(f, dtype=np.uint16)
        img = img.reshape(img_height, img_width)

    return img


def load_tiff_as_ndarray(img_path: str) -> np.ndarray:
    """
    Load a TIFF image using PIL and convert it to a numpy array.

    Parameters
    ----------
    img_path
        Path to the TIFF image file (.tif or .tiff)

    Returns
    -------
    np.ndarray
        2D numpy ndarray representing the image.
    """
    with Image.open(img_path) as img:
        return np.array(img)


def load_png_as_ndarray(img_path: str) -> np.ndarray:
    """
    Load a PNG image using PIL and convert it to a numpy array.

    Parameters
    ----------
    img_path
        Path to the PNG image file (.png)

    Returns
    -------
    np.ndarray
        2D numpy ndarray representing the image.
    """
    with Image.open(img_path) as img:
        return np.array(img)


def load_dicom_as_ndarray(img_path: str) -> np.ndarray:
    """
    Load a DICOM image using pydicom and convert it to a numpy array.

    Parameters
    ----------
    img_path
        Path to the DICOM image file (.dcm)

    Returns
    -------
    np.ndarray
        2D numpy ndarray representing the image.
    """
    dataset = pydicom.dcmread(img_path)
    return dataset.pixel_array


def load_image(img_path: str) -> np.ndarray:
    """
    Load an image and dispatch to the correct loader based on file extension.

    Parameters
    ----------
    img_path
        Path to the image file.

    Returns
    -------
    np.ndarray
        2D numpy ndarray representing the image.

    Notes
    -----
    Supported formats: ``.raw`` (with matching ``.xml``), ``.tif``/``.tiff``,
    ``.png``, and ``.dcm``.
    """
    ext = Path(img_path).suffix.lower()
    if ext == ".raw":
        return load_raw_as_ndarray(img_path)
    elif ext in [".tif", ".tiff"]:
        return load_tiff_as_ndarray(img_path)
    elif ext == ".png":
        return load_png_as_ndarray(img_path)
    elif ext == ".dcm":
        return load_dicom_as_ndarray(img_path)
    else:
        raise ValueError(f"Unsupported image format: {ext}")
