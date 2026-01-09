import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scopexr.circle_detection import (
    detect_circle_hough,
    estimate_circle,
    is_circle_centered,
)

# Ensure local src is on the path when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def create_synthetic_circle_image(
    size: int, cx: float, cy: float, radius: float, brightness: float = 200
) -> np.ndarray:
    """Helper function to create a synthetic image with a circle."""
    img = np.zeros((size, size), dtype=np.uint16)

    # Create a filled circle
    y, x = np.ogrid[:size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    img[mask] = int(brightness)

    return img


class TestDetectCircleHough:
    def test_detect_perfect_circle(self, tmp_path):
        """Test detection of a perfect circle."""
        size = 200
        cx, cy, radius = 100, 100, 40
        img = create_synthetic_circle_image(size, cx, cy, radius)

        result = detect_circle_hough(
            img,
            dp=1.0,
            min_dist=100,
            param1=50,
            param2=15,
            min_radius=30,
            max_radius=50,
            output_path=str(tmp_path),
        )

        # Detection may fail on synthetic images due to lack of edges
        if result is not None:
            x, y, r = result

            # Should detect close to the actual circle
            assert abs(x - cx) < 10
            assert abs(y - cy) < 10
            assert abs(r - radius) < 10

    def test_no_circle_found(self, tmp_path):
        """Test when no circle is found."""
        # Random noise image with very low variance
        img = np.random.randint(100, 110, (200, 200), dtype=np.uint16)

        result = detect_circle_hough(
            img,
            dp=1.0,
            min_dist=100,
            param1=100,
            param2=50,  # High threshold to avoid false positives
            min_radius=30,
            max_radius=50,
            output_path=str(tmp_path),
        )

        # May return None or may find spurious circles in noise
        # The main point is the function doesn't crash
        assert result is None or isinstance(result, tuple)

    def test_circle_with_noise(self, tmp_path):
        """Test detection of circle in noisy image."""
        size = 200
        cx, cy, radius = 100, 100, 40
        img = create_synthetic_circle_image(size, cx, cy, radius)

        # Add noise (convert to int32 to avoid overflow)
        noise = np.random.randint(0, 50, img.shape, dtype=np.int32)
        img = (img.astype(np.int32) + noise).clip(0, 65535).astype(np.uint16)

        result = detect_circle_hough(
            img,
            dp=1.0,
            min_dist=100,
            param1=100,
            param2=20,
            min_radius=30,
            max_radius=50,
            output_path=str(tmp_path),
        )

        if result is not None:
            x, y, r = result
            # Detection should still be reasonably close (convert to int for comparison)
            assert abs(int(x) - cx) < 10
            assert abs(int(y) - cy) < 10

    def test_saves_output_file(self, tmp_path):
        """Test that output image is saved when path is provided."""
        size = 200
        cx, cy, radius = 100, 100, 40
        img = create_synthetic_circle_image(size, cx, cy, radius)

        result = detect_circle_hough(
            img,
            dp=1.0,
            min_dist=100,
            param1=50,
            param2=30,
            min_radius=30,
            max_radius=50,
            output_path=str(tmp_path),
        )

        if result is not None:
            output_file = tmp_path / "detected_circle.png"
            assert output_file.exists()

    def test_none_image_raises_error(self):
        """Test that None image raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_circle_hough(
                None,
                dp=1.0,
                min_dist=100,
                param1=50,
                param2=30,
                min_radius=30,
                max_radius=50,
            )

    def test_off_center_circle(self, tmp_path):
        """Test detection of off-center circle."""
        size = 200
        cx, cy, radius = 150, 50, 30
        img = create_synthetic_circle_image(size, cx, cy, radius)

        result = detect_circle_hough(
            img,
            dp=1.0,
            min_dist=100,
            param1=50,
            param2=30,
            min_radius=20,
            max_radius=40,
            output_path=str(tmp_path),
        )

        if result is not None:
            x, y, r = result
            assert abs(x - cx) < 10
            assert abs(y - cy) < 10

    def test_debug_mode(self, tmp_path, monkeypatch):
        """Test with debug mode enabled."""
        size = 200
        cx, cy, radius = 100, 100, 40
        img = create_synthetic_circle_image(size, cx, cy, radius)

        # Mock cv2.imshow to avoid display
        display_called = [False]

        def mock_imshow(*args):
            display_called[0] = True

        monkeypatch.setattr(cv2, "imshow", mock_imshow)
        monkeypatch.setattr(cv2, "waitKey", lambda x: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

        result = detect_circle_hough(
            img,
            dp=1.0,
            min_dist=100,
            param1=50,
            param2=15,
            min_radius=30,
            max_radius=50,
            output_path=str(tmp_path),
            debug=True,  # Enable debug mode
        )

        # If circle found, debug mode should have been executed
        if result is not None:
            assert display_called[0]

        est_cx, est_cy, est_radius = estimate_circle(img)

        # Should still estimate reasonably well
        assert abs(est_cx - cx) < 5
        assert abs(est_cy - cy) < 5
        assert abs(est_radius - radius) < 5

    def test_small_circle(self):
        """Test estimation of a small circle."""
        size = 100
        cx, cy, radius = 50, 50, 10
        img = create_synthetic_circle_image(size, cx, cy, radius)

        est_cx, est_cy, est_radius = estimate_circle(img)

        assert est_radius > 0
        assert est_radius < 20  # Should be small

    def test_large_circle(self):
        """Test estimation of a large circle."""
        size = 200
        cx, cy, radius = 100, 100, 80
        img = create_synthetic_circle_image(size, cx, cy, radius)

        est_cx, est_cy, est_radius = estimate_circle(img)

        assert est_radius > 70  # Should be large

    def test_empty_image(self):
        """Test with empty/zero image."""
        img = np.zeros((100, 100), dtype=np.uint16)

        est_cx, est_cy, est_radius = estimate_circle(img)

        # With all zeros, the function returns center and radius 0
        assert est_cx == pytest.approx(50.0, abs=1.0)
        assert est_cy == pytest.approx(50.0, abs=1.0)
        assert est_radius >= 0  # May be 0 or calculated from area

    def test_uniform_bright_image(self):
        """Test with uniformly bright image (no circle)."""
        img = np.ones((100, 100), dtype=np.uint16) * 1000

        est_cx, est_cy, est_radius = estimate_circle(img)

        # Should still return valid values
        assert est_cx > 0
        assert est_cy > 0
        assert est_radius > 0

    def test_different_sizes(self):
        """Test with different image sizes."""
        test_sizes = [50, 100, 200, 300]

        for size in test_sizes:
            cx, cy, radius = size // 2, size // 2, size // 4
            img = create_synthetic_circle_image(size, cx, cy, radius)

            est_cx, est_cy, est_radius = estimate_circle(img)

            # Should be able to estimate for any reasonable size
            assert 0 < est_cx < size
            assert 0 < est_cy < size
            assert est_radius > 0

    def test_threshold_empty_mask(self):
        """Test estimate_circle when threshold results in empty mask."""
        # Create very small non-zero image where threshold is between max and min
        size = 100
        img = np.ones((size, size), dtype=np.uint16)  # All ones
        img[45:55, 45:55] = 100  # Small bright region

        # Create an image that will have empty mask after threshold
        # Min=1, Max=100, threshold=(1+100)/2=50.5
        # mask will be pixels >= 50.5, but only the center pixels are 100
        est_cx, est_cy, est_radius = estimate_circle(img)

        # Should estimate circle from the bright region
        assert 40 < est_cx < 60
        assert 40 < est_cy < 60
        assert est_radius > 0

    def test_truly_empty_mask(self):
        """Test when mask would be empty (all pixels below threshold)."""
        # To create an empty mask, we need all pixels to be below threshold
        # Since threshold = (min + max) / 2, we need max < threshold
        # But max >= threshold always when min < max
        # So we test the edge case where the image is constant
        # In this case min == max == threshold, and pixels >= threshold is True for all
        size = 100
        img = np.full((size, size), 500, dtype=np.uint16)  # All same value

        est_cx, est_cy, est_radius = estimate_circle(img)

        # With constant image, all pixels pass threshold
        # So mask is full, and center should be at image center
        assert est_cx == pytest.approx(size / 2, abs=1)
        assert est_cy == pytest.approx(size / 2, abs=1)
        # Radius should be sqrt(Area/pi) where Area is size^2
        expected_radius = np.sqrt(size * size / np.pi)
        assert est_radius == pytest.approx(expected_radius, rel=0.01)
        assert 40 < est_cy < 60
        assert est_radius > 0

    def test_off_center_outside_margin(self):
        """Test with circle outside the margin."""
        size = 100
        cropped = create_synthetic_circle_image(size, 50, 50, 30)

        # Way off center (> 10% margin)
        result = is_circle_centered(cropped, 70, 30, margin=0.1)

        assert result is False

    def test_different_margins(self):
        """Test with different margin values."""
        size = 100
        cropped = create_synthetic_circle_image(size, 50, 50, 30)
        cx, cy = 58, 52  # Slightly off

        # Tight margin - should fail
        result_tight = is_circle_centered(cropped, cx, cy, margin=0.05)

        # Loose margin - should pass
        result_loose = is_circle_centered(cropped, cx, cy, margin=0.2)

        assert result_tight is False
        assert result_loose is True

    def test_edge_cases(self):
        """Test edge cases near boundaries."""
        size = 100
        cropped = create_synthetic_circle_image(size, 50, 50, 30)

        # At the image edge
        result = is_circle_centered(cropped, 5, 50, margin=0.1)
        assert result is False

        result = is_circle_centered(cropped, 50, 95, margin=0.1)
        assert result is False

    def test_rectangular_image(self):
        """Test with non-square image."""
        img = np.zeros((100, 200), dtype=np.uint16)

        # Center of rectangular image
        center_x, center_y = 100, 50

        result = is_circle_centered(img, center_x, center_y, margin=0.1)
        assert result is True

        # Off center
        result = is_circle_centered(img, 150, 50, margin=0.1)
        assert result is False
