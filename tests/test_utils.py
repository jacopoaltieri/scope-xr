import sys
from pathlib import Path

import numpy as np
import pytest

from scopexr.utils import (
    eval_minimum_magnification,
    eval_minimum_radius,
    crop_square_roi,
    save_16bit_tiff,
    interpolate_nans_1d,
    suggest_os_angle,
    save_and_plot,
    background_percentile,
)

# Ensure local src is on the path when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestEvalMinimumMagnification:
    def test_basic_calculation(self):
        """Test basic magnification calculation."""
        a = 1.0  # 1mm focal spot
        n = 100  # 100 pixels
        p = 0.01  # 0.01mm pixel size

        result = eval_minimum_magnification(a, n, p)
        expected = (a + n * p) / a  # (1 + 1) / 1 = 2

        assert result == pytest.approx(expected)
        assert result == 2.0

    def test_different_values(self):
        """Test with different parameter combinations."""
        test_cases = [
            (0.5, 50, 0.02, 3.0),  # (0.5 + 1.0) / 0.5 = 3.0
            (2.0, 200, 0.01, 2.0),  # (2.0 + 2.0) / 2.0 = 2.0
            (1.5, 75, 0.02, 2.0),  # (1.5 + 1.5) / 1.5 = 2.0
        ]

        for a, n, p, expected in test_cases:
            result = eval_minimum_magnification(a, n, p)
            assert result == pytest.approx(expected)

    def test_edge_cases(self):
        """Test edge cases with small values."""
        # Very small focal spot
        result = eval_minimum_magnification(0.01, 100, 0.01)
        assert result > 100  # Should require high magnification

        # Very large focal spot
        result = eval_minimum_magnification(10.0, 100, 0.01)
        assert result < 2  # Should require low magnification


class TestEvalMinimumRadius:
    def test_basic_calculation(self):
        """Test basic radius calculation."""
        n = 100
        p = 0.01
        m = 2.0

        result = eval_minimum_radius(n, p, m)
        expected = (1 + n**2) * p / (2 * m)

        assert result == pytest.approx(expected)

    def test_different_values(self):
        """Test with different parameter combinations."""
        test_cases = [
            (50, 0.02, 1.5),
            (200, 0.01, 3.0),
            (10, 0.05, 1.0),
        ]

        for n, p, m in test_cases:
            result = eval_minimum_radius(n, p, m)
            expected = (1 + n**2) * p / (2 * m)
            assert result == pytest.approx(expected)
            assert result > 0


class TestCropSquareRoi:
    def test_basic_crop(self):
        """Test basic cropping functionality."""
        img = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        center = (50, 50)
        radius = 10

        cropped = crop_square_roi(img, center, radius, width_factor=1.5)

        expected_size = int(radius * 1.5) * 2
        assert cropped.shape[0] == expected_size
        assert cropped.shape[1] == expected_size

    def test_crop_near_edge(self):
        """Test cropping near image boundaries."""
        img = np.ones((100, 100), dtype=np.uint16)
        center = (10, 10)  # Near edge
        radius = 10

        cropped = crop_square_roi(img, center, radius, width_factor=2.0)

        # Should be clipped to image boundaries
        assert cropped.shape[0] <= 40
        assert cropped.shape[1] <= 40

    def test_crop_with_save(self, tmp_path):
        """Test cropping with saving to file."""
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        center = (50, 50)
        radius = 15

        cropped = crop_square_roi(
            img, center, radius, width_factor=1.5, output_path=str(tmp_path)
        )

        # Check that file was saved
        saved_file = tmp_path / "cropped.png"
        assert saved_file.exists()

        # Check cropped dimensions
        expected_size = int(radius * 1.5) * 2
        assert cropped.shape == (expected_size, expected_size)

    def test_different_width_factors(self):
        """Test with different width factors."""
        img = np.ones((100, 100), dtype=np.uint16)
        center = (50, 50)
        radius = 10

        for factor in [1.0, 1.5, 2.0, 3.0]:
            cropped = crop_square_roi(img, center, radius, width_factor=factor)
            expected_size = int(radius * factor) * 2
            assert cropped.shape == (expected_size, expected_size)


class TestSave16bitTiff:
    def test_save_normal_data(self, tmp_path):
        """Test saving normal data range."""
        data = np.random.rand(50, 50) * 100
        path = tmp_path / "test.tiff"

        save_16bit_tiff(data, str(path))

        assert path.exists()

    def test_save_constant_zero_image(self, tmp_path):
        """Test saving image with all zeros."""
        data = np.zeros((50, 50))
        path = tmp_path / "test_zeros.tiff"

        save_16bit_tiff(data, str(path))

        assert path.exists()

    def test_save_constant_nonzero_image(self, tmp_path):
        """Test saving image with constant non-zero value."""
        data = np.ones((50, 50)) * 42
        path = tmp_path / "test_const.tiff"

        save_16bit_tiff(data, str(path))

        assert path.exists()

    def test_data_range_scaling(self, tmp_path):
        """Test that data is properly scaled to 16-bit range."""
        data = np.array([[0, 100], [50, 75]], dtype=float)
        path = tmp_path / "test_scale.tiff"

        save_16bit_tiff(data, str(path))

        # Load and check that values are in uint16 range
        import tifffile

        loaded = tifffile.imread(str(path))
        assert loaded.dtype == np.uint16
        assert loaded.min() >= 0
        assert loaded.max() <= 65535


class TestInterpolateNans1d:
    def test_no_nans(self):
        """Test array without NaNs."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = interpolate_nans_1d(y)
        np.testing.assert_array_equal(result, y)

    def test_single_nan_middle(self):
        """Test interpolation of single NaN in the middle."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = interpolate_nans_1d(y)

        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 3.0  # Interpolated
        assert result[3] == 4.0
        assert result[4] == 5.0

    def test_multiple_nans(self):
        """Test interpolation of multiple NaNs."""
        y = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        result = interpolate_nans_1d(y)

        assert not np.any(np.isnan(result))
        assert result[0] == 1.0
        assert result[3] == 4.0
        assert result[4] == 5.0

    def test_all_nans(self):
        """Test array with all NaNs."""
        y = np.array([np.nan, np.nan, np.nan])
        result = interpolate_nans_1d(y)

        # Should return zeros
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_nans_at_edges(self):
        """Test NaNs at beginning and end."""
        y = np.array([np.nan, 2.0, 3.0, 4.0, np.nan])
        result = interpolate_nans_1d(y)

        # Edge NaNs should be extrapolated
        assert not np.any(np.isnan(result))
        assert result[1] == 2.0
        assert result[2] == 3.0
        assert result[3] == 4.0


class TestSuggestOsAngle:
    def test_basic_calculation(self):
        """Test basic angle suggestion."""
        p = 0.01  # pixel size
        n = 2  # oversampling factor
        r = 10.0  # radius

        result = suggest_os_angle(p, n, r)

        # Should return angle in degrees
        assert result > 0
        assert result < 180

    def test_different_parameters(self):
        """Test with different parameter combinations."""
        test_cases = [
            (0.01, 2, 10.0),
            (0.02, 3, 15.0),
            (0.005, 4, 20.0),
        ]

        for p, n, r in test_cases:
            result = suggest_os_angle(p, n, r)

            # Verify it's a reasonable angle
            assert result > 0
            assert result < 180

            # Larger radius should give smaller angle
            if r > 10:
                smaller_result = suggest_os_angle(p, n, 5.0)
                assert result < smaller_result


class TestSaveAndPlot:
    def test_save_without_plot(self, tmp_path):
        """Test saving array without plotting."""
        arr = np.random.rand(50, 50) * 100
        name = "test_image"

        result_path = save_and_plot(name, arr, str(tmp_path))

        assert Path(result_path).exists()
        assert "test_image" in result_path
        assert result_path.endswith(".tiff")

    def test_save_with_suffix(self, tmp_path):
        """Test saving with custom suffix."""
        arr = np.random.rand(50, 50) * 100
        name = "test_image"
        suffix = "_processed"

        result_path = save_and_plot(name, arr, str(tmp_path), suffix=suffix)

        assert Path(result_path).exists()
        assert "_processed" in result_path

    def test_save_with_tiff_extension(self, tmp_path):
        """Test saving when name already has .tiff extension."""
        arr = np.random.rand(50, 50) * 100
        name = "test_image.tiff"

        result_path = save_and_plot(name, arr, str(tmp_path))

        assert Path(result_path).exists()
        # Should not double the extension
        assert result_path.count(".tiff") == 1

    def test_save_with_plot_func(self, tmp_path):
        """Test saving with plotting function."""
        arr = np.random.rand(50, 50) * 100
        name = "test_image"

        # Mock plotting function
        plot_called = [False]

        def mock_plot(arr, out_dir, show):
            plot_called[0] = True

        result_path = save_and_plot(
            name, arr, str(tmp_path), plot_func=mock_plot, show_plots=False
        )

        assert Path(result_path).exists()
        assert plot_called[0]  # Verify plot function was called


class TestBackgroundPercentile:
    def test_basic_calculation(self):
        """Test basic background percentile calculation."""
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = background_percentile(profile, low_frac=0.2)

        # 20th percentile of [1,2,3,4,5] is 1.8
        # Values <= 1.8 are [1.0]
        # Mean of [1.0] is 1.0
        assert result == pytest.approx(1.0)

    def test_default_low_frac(self):
        """Test with default low_frac=0.15."""
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = background_percentile(profile)

        # Default low_frac=0.15 means 15th percentile
        assert isinstance(result, float)
        assert result > 0
        threshold = np.percentile(profile, 15)
        expected = float(np.mean(profile[profile <= threshold]))
        assert result == pytest.approx(expected)

    def test_noisy_profile(self):
        """Test with noisy profile containing signal and background."""
        np.random.seed(42)
        # Low values (noise/background) and high values (signal)
        background = np.random.rand(50) * 10
        signal = np.random.rand(50) * 100 + 50
        profile = np.concatenate([background, signal])
        np.random.shuffle(profile)

        result = background_percentile(profile, low_frac=0.25)

        # Should estimate background (lower values)
        assert result < np.median(profile)

    def test_uniform_profile(self):
        """Test with uniform profile."""
        profile = np.ones(100) * 5.0

        result = background_percentile(profile)

        # All values are the same, so result should equal that value
        assert result == pytest.approx(5.0)

    def test_single_value(self):
        """Test with single value array."""
        profile = np.array([10.0])

        result = background_percentile(profile)

        assert result == pytest.approx(10.0)

    def test_two_values(self):
        """Test with two values."""
        profile = np.array([1.0, 10.0])

        result = background_percentile(profile, low_frac=0.5)

        # 50th percentile of [1, 10] is 5.5
        # Values <= 5.5 are [1.0]
        # Mean is 1.0
        assert result == pytest.approx(1.0)

    def test_all_same_values_different_fraction(self):
        """Test with all same values using different fractions."""
        profile = np.ones(100) * 7.5

        for frac in [0.1, 0.25, 0.5, 0.75]:
            result = background_percentile(profile, low_frac=frac)
            assert result == pytest.approx(7.5)

    def test_extremes_fraction(self):
        """Test with extreme fraction values."""
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Very low fraction (just the minimum)
        result_low = background_percentile(profile, low_frac=0.01)
        assert result_low <= np.min(profile) + 0.1

        # Higher fraction
        result_high = background_percentile(profile, low_frac=0.6)
        assert result_high > result_low

    def test_negative_values(self):
        """Test with negative values in profile."""
        profile = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])

        result = background_percentile(profile, low_frac=0.4)

        # Should still work with negative values
        assert isinstance(result, float)
        # Result should be negative or close to zero
        assert result < 0 or result == pytest.approx(0.0, abs=1.0)

    def test_return_type(self):
        """Test that return type is float (not numpy scalar)."""
        profile = np.array([1.0, 2.0, 3.0])

        result = background_percentile(profile)

        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)

    def test_different_array_types(self):
        """Test with different array input types."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        # List input
        result_list = background_percentile(values)

        # NumPy array
        result_array = background_percentile(np.array(values))

        # Results should be the same
        assert result_list == pytest.approx(result_array)
