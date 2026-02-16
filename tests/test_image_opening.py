import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from scopexr.image_opening import (
    load_raw_as_ndarray,
    load_tiff_as_ndarray,
    load_png_as_ndarray,
    load_dicom_as_ndarray,
    load_image,
)

# Ensure local src is on the path when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _create_test_xml(xml_path: Path, width: int, height: int) -> None:
    """Helper function to create a test XML metadata file."""
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<root>
    <frame>
        <imgWidth>{width}</imgWidth>
        <imgHeight>{height}</imgHeight>
    </frame>
</root>"""
    xml_path.write_text(xml_content)


def _create_test_raw(
    raw_path: Path, width: int, height: int, data: np.ndarray = None
) -> np.ndarray:
    """Helper function to create a test RAW image file."""
    if data is None:
        # Create a simple gradient pattern
        data = np.arange(width * height, dtype=np.uint16).reshape(height, width)
    data.tofile(raw_path)
    return data


def _create_test_tiff(
    tiff_path: Path, width: int, height: int, data: np.ndarray = None
) -> np.ndarray:
    """Helper function to create a test TIFF image file."""
    if data is None:
        data = np.arange(width * height, dtype=np.uint16).reshape(height, width)
    img = Image.fromarray(data)
    img.save(tiff_path)
    return data


def _create_test_png(
    png_path: Path, width: int, height: int, data: np.ndarray = None
) -> np.ndarray:
    """Helper function to create a test PNG image file."""
    if data is None:
        # PNG typically uses 8-bit or 16-bit data
        data = (np.arange(width * height) % 256).astype(np.uint8).reshape(height, width)
    img = Image.fromarray(data)
    img.save(png_path)
    return data


def _create_test_dicom(
    dcm_path: Path, width: int, height: int, data: np.ndarray = None
) -> np.ndarray:
    """Helper function to create a test DICOM image file."""
    if data is None:
        data = np.arange(width * height, dtype=np.uint16).reshape(height, width)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(
        dcm_path,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    dataset.Rows = height
    dataset.Columns = width
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = data.tobytes()

    pydicom.dcmwrite(
        dcm_path,
        dataset,
        implicit_vr=False,
        little_endian=True,
        enforce_file_format=True,
    )
    return data


class TestLoadRawAsNdarray:
    def test_load_raw_basic(self, tmp_path):
        """Test basic loading of RAW image with XML metadata."""
        width, height = 100, 80
        raw_path = tmp_path / "test_image.raw"
        xml_path = tmp_path / "test_image.xml"

        expected_data = _create_test_raw(raw_path, width, height)
        _create_test_xml(xml_path, width, height)

        result = load_raw_as_ndarray(str(raw_path))

        assert result.shape == (height, width)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, expected_data)

    def test_load_raw_different_dimensions(self, tmp_path):
        """Test loading RAW images with different dimensions."""
        test_cases = [(50, 50), (200, 100), (10, 500)]

        for width, height in test_cases:
            raw_path = tmp_path / f"test_{width}x{height}.raw"
            xml_path = tmp_path / f"test_{width}x{height}.xml"

            expected_data = _create_test_raw(raw_path, width, height)
            _create_test_xml(xml_path, width, height)

            result = load_raw_as_ndarray(str(raw_path))

            assert result.shape == (height, width)
            np.testing.assert_array_equal(result, expected_data)

    def test_load_raw_custom_data(self, tmp_path):
        """Test loading RAW image with custom data pattern."""
        width, height = 50, 40
        raw_path = tmp_path / "test_custom.raw"
        xml_path = tmp_path / "test_custom.xml"

        # Create custom data pattern
        custom_data = np.random.randint(0, 65535, size=(height, width), dtype=np.uint16)
        _create_test_raw(raw_path, width, height, custom_data)
        _create_test_xml(xml_path, width, height)

        result = load_raw_as_ndarray(str(raw_path))

        np.testing.assert_array_equal(result, custom_data)

    def test_load_raw_missing_xml(self, tmp_path):
        """Test that FileNotFoundError is raised when XML metadata is missing."""
        raw_path = tmp_path / "test_no_xml.raw"
        width, height = 100, 80
        _create_test_raw(raw_path, width, height)

        with pytest.raises(FileNotFoundError, match="Metadata XML not found"):
            load_raw_as_ndarray(str(raw_path))

    def test_load_raw_nonexistent_file(self):
        """Test that appropriate error is raised for non-existent RAW file."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_raw_as_ndarray("nonexistent.raw")


class TestLoadTiffAsNdarray:
    def test_load_tiff_basic(self, tmp_path):
        """Test basic loading of TIFF image."""
        width, height = 100, 80
        tiff_path = tmp_path / "test_image.tif"

        expected_data = _create_test_tiff(tiff_path, width, height)
        result = load_tiff_as_ndarray(str(tiff_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_tiff_extension(self, tmp_path):
        """Test loading TIFF with .tiff extension."""
        width, height = 50, 50
        tiff_path = tmp_path / "test_image.tiff"

        expected_data = _create_test_tiff(tiff_path, width, height)
        result = load_tiff_as_ndarray(str(tiff_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_tiff_different_dimensions(self, tmp_path):
        """Test loading TIFF images with different dimensions."""
        test_cases = [(200, 150), (10, 10), (500, 100)]

        for width, height in test_cases:
            tiff_path = tmp_path / f"test_{width}x{height}.tif"
            expected_data = _create_test_tiff(tiff_path, width, height)
            result = load_tiff_as_ndarray(str(tiff_path))

            assert result.shape == (height, width)
            np.testing.assert_array_equal(result, expected_data)

    def test_load_tiff_nonexistent_file(self):
        """Test that appropriate error is raised for non-existent TIFF file."""
        with pytest.raises(FileNotFoundError):
            load_tiff_as_ndarray("nonexistent.tif")


class TestLoadPngAsNdarray:
    def test_load_png_basic(self, tmp_path):
        """Test basic loading of PNG image."""
        width, height = 100, 80
        png_path = tmp_path / "test_image.png"

        expected_data = _create_test_png(png_path, width, height)
        result = load_png_as_ndarray(str(png_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_png_different_dimensions(self, tmp_path):
        """Test loading PNG images with different dimensions."""
        test_cases = [(200, 150), (10, 10), (500, 100)]

        for width, height in test_cases:
            png_path = tmp_path / f"test_{width}x{height}.png"
            expected_data = _create_test_png(png_path, width, height)
            result = load_png_as_ndarray(str(png_path))

            assert result.shape == (height, width)
            np.testing.assert_array_equal(result, expected_data)

    def test_load_png_nonexistent_file(self):
        """Test that appropriate error is raised for non-existent PNG file."""
        with pytest.raises(FileNotFoundError):
            load_png_as_ndarray("nonexistent.png")


class TestLoadDicomAsNdarray:
    def test_load_dicom_basic(self, tmp_path):
        """Test basic loading of DICOM image."""
        width, height = 64, 48
        dcm_path = tmp_path / "test_image.dcm"

        expected_data = _create_test_dicom(dcm_path, width, height)
        result = load_dicom_as_ndarray(str(dcm_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_dicom_different_dimensions(self, tmp_path):
        """Test loading DICOM images with different dimensions."""
        test_cases = [(16, 16), (128, 32), (32, 128)]

        for width, height in test_cases:
            dcm_path = tmp_path / f"test_{width}x{height}.dcm"
            expected_data = _create_test_dicom(dcm_path, width, height)
            result = load_dicom_as_ndarray(str(dcm_path))

            assert result.shape == (height, width)
            np.testing.assert_array_equal(result, expected_data)

    def test_load_dicom_nonexistent_file(self):
        """Test that appropriate error is raised for non-existent DICOM file."""
        with pytest.raises(FileNotFoundError):
            load_dicom_as_ndarray("nonexistent.dcm")


class TestLoadImage:
    def test_load_image_raw(self, tmp_path):
        """Test load_image dispatches correctly for RAW files."""
        width, height = 100, 80
        raw_path = tmp_path / "test_image.raw"
        xml_path = tmp_path / "test_image.xml"

        expected_data = _create_test_raw(raw_path, width, height)
        _create_test_xml(xml_path, width, height)

        result = load_image(str(raw_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_image_tif(self, tmp_path):
        """Test load_image dispatches correctly for .tif files."""
        width, height = 100, 80
        tiff_path = tmp_path / "test_image.tif"

        expected_data = _create_test_tiff(tiff_path, width, height)
        result = load_image(str(tiff_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_image_tiff(self, tmp_path):
        """Test load_image dispatches correctly for .tiff files."""
        width, height = 100, 80
        tiff_path = tmp_path / "test_image.tiff"

        expected_data = _create_test_tiff(tiff_path, width, height)
        result = load_image(str(tiff_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_image_png(self, tmp_path):
        """Test load_image dispatches correctly for PNG files."""
        width, height = 100, 80
        png_path = tmp_path / "test_image.png"

        expected_data = _create_test_png(png_path, width, height)
        result = load_image(str(png_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_image_dcm(self, tmp_path):
        """Test load_image dispatches correctly for DICOM files."""
        width, height = 64, 48
        dcm_path = tmp_path / "test_image.dcm"

        expected_data = _create_test_dicom(dcm_path, width, height)
        result = load_image(str(dcm_path))

        assert result.shape == (height, width)
        np.testing.assert_array_equal(result, expected_data)

    def test_load_image_case_insensitive(self, tmp_path):
        """Test that file extensions are case-insensitive."""
        test_cases = [
            ("test.RAW", "raw"),
            ("test.TIF", "tiff"),
            ("test.TIFF", "tiff"),
            ("test.PNG", "png"),
            ("test.Png", "png"),
            ("test.DCM", "dicom"),
        ]

        for filename, file_type in test_cases:
            if file_type == "raw":
                raw_path = tmp_path / filename
                xml_path = tmp_path / (filename[:-4] + ".xml")
                _create_test_raw(raw_path, 50, 50)
                _create_test_xml(xml_path, 50, 50)
                result = load_image(str(raw_path))
            elif file_type == "tiff":
                tiff_path = tmp_path / filename
                _create_test_tiff(tiff_path, 50, 50)
                result = load_image(str(tiff_path))
            elif file_type == "png":
                png_path = tmp_path / filename
                _create_test_png(png_path, 50, 50)
                result = load_image(str(png_path))
            elif file_type == "dicom":
                dcm_path = tmp_path / filename
                _create_test_dicom(dcm_path, 50, 50)
                result = load_image(str(dcm_path))

            assert result.shape == (50, 50)

    def test_load_image_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported file formats."""
        unsupported_files = ["test.jpg", "test.bmp", "test.gif", "test.txt"]

        for filename in unsupported_files:
            file_path = tmp_path / filename
            file_path.write_text("dummy content")

            with pytest.raises(ValueError, match="Unsupported image format"):
                load_image(str(file_path))

    def test_load_image_no_extension(self, tmp_path):
        """Test that ValueError is raised for files without extension."""
        file_path = tmp_path / "test_no_ext"
        file_path.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported image format"):
            load_image(str(file_path))
