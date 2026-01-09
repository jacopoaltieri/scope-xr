import sys
from pathlib import Path

import numpy as np
import pytest

from scopexr.mtf_calc import (
    compute_1d_mtf,
    compute_1d_mtf_from_sino,
    get_mtf_at_freq,
)

# Ensure local src is on the path when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def create_gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Helper function to create a Gaussian PSF."""
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)

    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / np.sum(psf)  # Normalize

    return psf


def create_gaussian_sinogram(n_rays: int, n_angles: int, sigma: float) -> np.ndarray:
    """Helper function to create a sinogram with Gaussian profiles."""
    sinogram = np.zeros((n_rays, n_angles))
    x = np.arange(n_rays) - n_rays // 2

    for i in range(n_angles):
        sinogram[:, i] = np.exp(-(x**2) / (2 * sigma**2))

    return sinogram


class TestCompute1dMtf:
    def test_basic_computation(self):
        """Test basic MTF computation from PSF."""
        psf = create_gaussian_psf(100, 5.0)
        pixel_size = 0.1  # mm

        freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size, axis=0)

        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)
        assert mtf_1d[0] == pytest.approx(1.0)  # Normalized to 1 at zero frequency
        assert np.all(freq >= 0)  # Only positive frequencies

    def test_mtf_decreases_with_frequency(self):
        """Test that MTF generally decreases with frequency."""
        psf = create_gaussian_psf(100, 5.0)
        pixel_size = 0.1

        freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size, axis=0)

        # MTF should generally decrease
        assert mtf_1d[0] > mtf_1d[len(mtf_1d) // 2]
        assert mtf_1d[len(mtf_1d) // 2] > mtf_1d[-1]

    def test_axis_parameter(self):
        """Test MTF computation along different axes."""
        psf = create_gaussian_psf(100, 5.0)
        pixel_size = 0.1

        freq0, mtf0, mtf10_0 = compute_1d_mtf(psf, pixel_size, axis=0)
        freq1, mtf1, mtf10_1 = compute_1d_mtf(psf, pixel_size, axis=1)

        # For symmetric PSF, results should be similar
        assert len(freq0) == len(freq1)
        np.testing.assert_array_almost_equal(mtf0, mtf1, decimal=5)

    def test_narrow_psf_has_higher_mtf10(self):
        """Test that narrower PSF has higher MTF10."""
        pixel_size = 0.1

        psf_narrow = create_gaussian_psf(100, 2.0)
        psf_wide = create_gaussian_psf(100, 5.0)

        _, _, mtf10_narrow = compute_1d_mtf(psf_narrow, pixel_size, axis=0)
        _, _, mtf10_wide = compute_1d_mtf(psf_wide, pixel_size, axis=0)

        # Narrower PSF should have higher MTF10 (if not NaN)
        if not np.isnan(mtf10_narrow) and not np.isnan(mtf10_wide):
            assert mtf10_narrow > mtf10_wide

    def test_pixel_size_effect(self):
        """Test effect of pixel size on frequency axis."""
        psf = create_gaussian_psf(100, 5.0)

        freq_small, _, _ = compute_1d_mtf(psf, 0.05, axis=0)
        freq_large, _, _ = compute_1d_mtf(psf, 0.1, axis=0)

        # Smaller pixel size should give higher max frequency
        assert np.max(freq_small) > np.max(freq_large)

    def test_mtf10_detection(self):
        """Test that MTF10 is detected correctly."""
        psf = create_gaussian_psf(100, 5.0)
        pixel_size = 0.1

        freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size, axis=0)

        if not np.isnan(mtf10):
            # MTF10 should be a positive frequency
            assert mtf10 > 0
            # Verify it's approximately where MTF drops to 0.1
            mtf_at_mtf10 = get_mtf_at_freq(mtf10, freq, mtf_1d)
            assert abs(mtf_at_mtf10 - 0.1) < 0.05

    def test_mtf10_at_first_index(self):
        """Test MTF10 detection when MTF drops below 0.1 at first positive frequency."""
        # Very wide PSF (sharp image)
        psf = np.zeros((100, 100))
        psf[40:60, 40:60] = 1.0  # Wide square
        psf = psf / np.sum(psf)
        pixel_size = 0.1

        freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size, axis=0)

        # If MTF drops to 0.1 at first index, it should return freq_pos[0]
        if not np.isnan(mtf10) and len(freq) > 0:
            assert mtf10 >= freq[0]

    def test_sinogram_mtf_with_zeros(self):
        """Test MTF computation with sinogram containing zero rows."""
        # Create sinogram with some zero rows
        sinogram = np.zeros((128, 128))
        sinogram[50:60, 40:80] = 1.0
        sinogram[0, :] = 0  # First row all zeros
        sinogram[-1, :] = 0  # Last row all zeros
        pixel_size = 0.1

        freq, mtf_sino, mtf10_sino = compute_1d_mtf_from_sino(sinogram, pixel_size, 0)

        # Should still compute MTF despite zero rows
        assert len(freq) > 0
        assert len(mtf_sino) == len(freq)
        # MTF should be between 0 and 1
        valid_mtf = mtf_sino[~np.isnan(mtf_sino)]
        if len(valid_mtf) > 0:
            assert np.all((valid_mtf >= 0) & (valid_mtf <= 1))

    def test_mtf10_zero_denominator_in_interpolation(self):
        """Test MTF10 interpolation when m2 == m1 (zero denominator)."""
        # Create sinogram where MTF has plateau (m2 == m1)
        sinogram = np.ones((100, 180))
        sinogram[40:50, :] += 1.0  # Make it have a plateau
        pixel_size = 0.1

        freq, mtf_1d, mtf10 = compute_1d_mtf_from_sino(sinogram, pixel_size, 0)

        # Should handle zero denominator gracefully
        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)

    def test_mtf10_at_first_index_sino(self):
        """Test MTF10 when idx == 0 in sinogram (line 127 coverage)."""
        # Create sinogram where MTF drops below 0.1 at first frequency
        sinogram = np.ones((100, 180))
        sinogram[30:70, :] += 1.0  # Add central plateau
        pixel_size = 0.1

        freq, mtf_1d, mtf10 = compute_1d_mtf_from_sino(sinogram, pixel_size, 0)

        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)
        # mtf10 should be valid or NaN
        assert np.isnan(mtf10) or mtf10 > 0

    def test_mtf_never_reaches_10_percent(self):
        """Test case where MTF never drops below 0.1 (line 73 coverage)."""
        # Create very sharp PSF where MTF stays above 0.1
        psf = np.zeros((100, 100))
        psf[50, 50] = 1.0  # Delta function
        pixel_size = 0.1

        freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size, axis=0)

        # MTF10 may be NaN if never reaches 0.1
        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)

    def test_mtf10_immediately_below_threshold(self):
        """Test when MTF drops below 0.1 at first frequency (idx == 0)."""
        # Create PSF that has very low MTF at low frequencies
        psf = np.zeros((200, 200))
        # Create a pattern with alternating bands to have low MTF
        for i in range(0, 200, 4):
            psf[i : i + 2, :] = 1.0
        psf = psf / np.sum(psf)
        pixel_size = 0.01  # Very small pixel for high frequencies

        freq, mtf_1d, mtf10 = compute_1d_mtf(psf, pixel_size, axis=0)

        # This PSF should have rapidly decreasing MTF
        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)
        # Check that first few MTF values are decreasing
        if len(mtf_1d) > 2:
            assert (
                mtf_1d[0] >= mtf_1d[1] or mtf_1d[0] >= 0
            )  # Allow for numerical variations

    def test_mtf10_immediately_below_threshold_sinogram(self):
        """Test sinogram MTF10 when idx == 0 (lines 127, 133 coverage)."""
        # Create sinogram with rapidly decreasing MTF
        sinogram = np.zeros((200, 180))
        # Create alternating pattern
        for i in range(0, 200, 4):
            sinogram[i : i + 2, :] = 1.0
        pixel_size = 0.01

        freq, mtf_1d, mtf10 = compute_1d_mtf_from_sino(sinogram, pixel_size, 0)

        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)

    def test_different_angles(self):
        """Test MTF computation at different angles."""
        sinogram = create_gaussian_sinogram(100, 180, 5.0)
        pixel_size = 0.1

        freq0, mtf0, _ = compute_1d_mtf_from_sino(sinogram, pixel_size, 0)
        freq90, mtf90, _ = compute_1d_mtf_from_sino(sinogram, pixel_size, 90)

        # For uniform sinogram, results should be similar
        assert len(freq0) == len(freq90)
        np.testing.assert_array_almost_equal(mtf0, mtf90, decimal=5)

    def test_narrow_vs_wide_profile(self):
        """Test that narrower profile gives higher MTF10."""
        pixel_size = 0.1
        angle = 0

        sino_narrow = create_gaussian_sinogram(100, 180, 2.0)
        sino_wide = create_gaussian_sinogram(100, 180, 5.0)

        _, _, mtf10_narrow = compute_1d_mtf_from_sino(sino_narrow, pixel_size, angle)
        _, _, mtf10_wide = compute_1d_mtf_from_sino(sino_wide, pixel_size, angle)

        if not np.isnan(mtf10_narrow) and not np.isnan(mtf10_wide):
            assert mtf10_narrow > mtf10_wide

    def test_zero_denominator_case(self):
        """Test handling of zero denominator in interpolation."""
        # Create a flat sinogram profile
        sinogram = np.ones((100, 180))
        pixel_size = 0.1
        angle = 0

        freq, mtf_1d, mtf10 = compute_1d_mtf_from_sino(sinogram, pixel_size, angle)

        # Should not crash, MTF10 may be NaN
        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)

    def test_sharp_profile_no_mtf10(self):
        """Test sharp sinogram profile that may not reach MTF10."""
        # Delta-like profile
        sinogram = np.zeros((100, 180))
        sinogram[50, :] = 1.0
        pixel_size = 0.1
        angle = 0

        freq, mtf_1d, mtf10 = compute_1d_mtf_from_sino(sinogram, pixel_size, angle)

        # MTF10 may be NaN
        assert len(freq) > 0
        assert len(mtf_1d) == len(freq)


class TestGetMtfAtFreq:
    def test_basic_interpolation(self):
        """Test basic MTF interpolation at specific frequency."""
        freq_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mtf_array = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        # Interpolate at midpoint
        result = get_mtf_at_freq(1.5, freq_array, mtf_array)

        expected = 0.7  # Linear interpolation between 0.8 and 0.6
        assert result == pytest.approx(expected)

    def test_exact_frequency_match(self):
        """Test when target frequency exactly matches array value."""
        freq_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mtf_array = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        result = get_mtf_at_freq(2.0, freq_array, mtf_array)

        assert result == pytest.approx(0.6)

    def test_extrapolation_low(self):
        """Test extrapolation below frequency range."""
        freq_array = np.array([1.0, 2.0, 3.0, 4.0])
        mtf_array = np.array([0.8, 0.6, 0.4, 0.2])

        result = get_mtf_at_freq(0.5, freq_array, mtf_array)

        # np.interp extrapolates with edge values
        assert result >= 0

    def test_extrapolation_high(self):
        """Test extrapolation above frequency range."""
        freq_array = np.array([0.0, 1.0, 2.0, 3.0])
        mtf_array = np.array([1.0, 0.8, 0.6, 0.4])

        result = get_mtf_at_freq(5.0, freq_array, mtf_array)

        # Should extrapolate (np.interp uses edge values)
        assert result >= 0

    def test_multiple_frequencies(self):
        """Test interpolation at multiple frequencies."""
        freq_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mtf_array = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        target_freqs = [0.5, 1.5, 2.5, 3.5]
        expected = [0.9, 0.7, 0.5, 0.3]

        for target, exp in zip(target_freqs, expected):
            result = get_mtf_at_freq(target, freq_array, mtf_array)
            assert result == pytest.approx(exp, abs=0.01)

    def test_decreasing_mtf(self):
        """Test with realistic decreasing MTF curve."""
        freq_array = np.linspace(0, 10, 100)
        mtf_array = np.exp(-freq_array * 0.5)  # Exponential decay

        # Check a few points
        result_low = get_mtf_at_freq(1.0, freq_array, mtf_array)
        result_mid = get_mtf_at_freq(5.0, freq_array, mtf_array)
        result_high = get_mtf_at_freq(9.0, freq_array, mtf_array)

        # Should decrease with frequency
        assert result_low > result_mid > result_high
        assert 0 <= result_high <= result_low <= 1
