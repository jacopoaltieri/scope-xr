import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


def load_raw_as_ndarray(img_path: str) -> np.ndarray:
    """Load a raw image as a numpy ndarray using metadata from the corresponding XML file."""
    xml_path = os.path.splitext(img_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Metadata XML not found: '{xml_path}'")

    # Parse XML and get width and height
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame = root.find("frame")
    img_width = int(frame.find("imgWidth").text)
    img_height = int(frame.find("imgHeight").text)

    # Read and reshape raw data
    with open(img_path, "rb") as f:
        img = np.fromfile(f, dtype=np.uint16)
        img = img.reshape(img_height, img_width)
    return img


def load_tiff_as_ndarray(img_path: str) -> np.ndarray:
    """Load a TIFF image using PIL and convert it to a numpy array."""
    with Image.open(img_path) as img:
        return np.array(img)


def load_png_as_ndarray(img_path: str) -> np.ndarray:
    """Load a PNG image using PIL and convert it to a numpy array."""
    with Image.open(img_path) as img:
        return np.array(img)


def load_image(img_path: str) -> np.ndarray:
    """Load an image and dispatch to the correct loader based on file extension."""
    ext = os.path.splitext(img_path)[1].lower()
    if ext == ".raw":
        return load_raw_as_ndarray(img_path)
    elif ext in [".tif", ".tiff"]:
        return load_tiff_as_ndarray(img_path)
    elif ext == ".png":
        return load_png_as_ndarray(img_path)
    else:
        raise ValueError(f"Unsupported image format: {ext}")
