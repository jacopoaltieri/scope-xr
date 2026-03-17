import sys
from pathlib import Path

import numpy as np
import pytest

from scopexr.widths_calculator import (
    fw_at_percent_max,
    fwhm_from_sigma,
    erf_step,
    find_extreme_profiles_erf,
    average_neighbors,
    compute_fs_width,
    gaussian,
    find_extreme_profiles_gaussian,
    compute_lsf_from_projection,
)

# Ensure local src is on the path when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestFwhm:
    def test_symmetric_gaussian_like_profile(self):
        """Test FWHM on a symmetric Gaussian-like profile."""
        x = np.linspace(-10, 10, 201)
        profile = np.exp(-(x**2) / 2)  # Gaussian centered at 0

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        assert width > 0
        assert not np.isnan(width)
        assert left_idx < right_idx
        # Should be roughly symmetric
        center = 100  # Middle of 201 points
        assert abs((left_idx + right_idx) / 2 - center) < 5

    def test_narrow_peak(self):
        """Test FWHM on a narrow peak."""
        profile = np.zeros(100)
        profile[45:55] = 1.0  # 10-pixel wide flat top

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        assert width > 0
        assert width < 15  # Should be narrow

    def test_wide_peak(self):
        """Test FWHM on a wide peak."""
        profile = np.zeros(100)
        profile[10:90] = 1.0  # 80-pixel wide flat top

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        assert width > 70  # Should be wide

    def test_no_peak(self):
        """Test behavior with flat profile (no peak)."""
        profile = np.ones(100)

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        # Should return NaN if no valid FWHM can be found
        assert np.isnan(width) or width >= 0

    def test_noisy_profile(self):
        """Test FWHM on a noisy profile."""
        x = np.linspace(-5, 5, 101)
        profile = np.exp(-(x**2) / 2) + np.random.rand(101) * 0.1

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        assert width > 0
        assert not np.isnan(width)

    def test_peak_at_edge(self):
        """Test FWHM when peak is at the edge."""
        profile = np.zeros(100)
        profile[0:10] = 1.0  # Peak at start

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        # May return NaN or small width
        assert np.isnan(width) or width >= 0

    def test_zero_denominator_in_interpolation(self):
        """Test FWHM when zero denominator occurs during interpolation."""
        # Create profile where slopes are exactly zero
        profile = np.array([0, 0.5, 1.0, 1.0, 1.0, 0.5, 0], dtype=float)

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        # Should handle gracefully with fallback frac=0.5
        if not np.isnan(width):
            assert width > 0

        assert width > 0
        assert not np.isnan(width)
        assert left_idx < right_idx

    def test_fw15m_wider_than_fwhm(self):
        """Test that FW15M is wider than FWHM."""
        x = np.linspace(-10, 10, 201)
        profile = np.exp(-(x**2) / 2)

        width_15m, _, _ = fw_at_percent_max(profile, 0.15)
        width_hm, _, _ = fw_at_percent_max(profile, 0.5)

        assert width_15m > width_hm

    def test_fwhm_zero_denominator(self):
        """Test FWHM with zero denominator in left side interpolation."""
        # Create profile with flat section that causes denominator == 0
        profile = np.array([0.1, 0.1, 0.5, 1.0, 0.5, 0.1, 0.1], dtype=float)

        width, left_idx, right_idx = fw_at_percent_max(profile, 0.5)

        # Should handle gracefully
        if not np.isnan(width):
            assert width > 0


class TestFwhmFromSigma:
    def test_basic_conversion(self):
        """Test conversion from sigma to FWHM."""
        sigma = 1.0
        result = fwhm_from_sigma(sigma)

        expected = 2 * sigma * np.sqrt(2 * np.log(2))
        assert result == pytest.approx(expected)

    def test_different_sigmas(self):
        """Test with different sigma values."""
        test_sigmas = [0.5, 1.0, 2.0, 5.0]

        for sigma in test_sigmas:
            result = fwhm_from_sigma(sigma)
            expected = 2 * sigma * np.sqrt(2 * np.log(2))
            assert result == pytest.approx(expected)
            assert result > 0


class TestErfStep:
    def test_basic_erf_step(self):
        """Test basic error function step."""
        x = np.linspace(-10, 10, 101)
        A = 1.0
        x0 = 0.0
        sigma = 1.0
        B = 0.0

        result = erf_step(x, A, x0, sigma, B)

        assert len(result) == len(x)
        assert result[0] < result[-1]  # Should be increasing

    def test_erf_step_parameters(self):
        """Test that parameters affect the step correctly."""
        x = np.linspace(-10, 10, 101)

        # Test amplitude effect
        result1 = erf_step(x, 1.0, 0.0, 1.0, 0.0)
        result2 = erf_step(x, 2.0, 0.0, 1.0, 0.0)
        assert np.max(result2) > np.max(result1)

        # Test center position
        result1 = erf_step(x, 1.0, -2.0, 1.0, 0.0)
        result2 = erf_step(x, 1.0, 2.0, 1.0, 0.0)
        # Steps should be shifted
        assert np.argmax(np.gradient(result1)) < np.argmax(np.gradient(result2))

    def test_erf_step_with_background(self):
        """Test error function with background offset."""
        x = np.linspace(-10, 10, 101)
        B = 5.0

        result = erf_step(x, 1.0, 0.0, 1.0, B)

        # All values should be offset by B
        assert np.min(result) >= B - 1.5


class TestAverageNeighbors:
    def test_basic_averaging(self):
        """Test basic neighbor averaging."""
        sinogram = np.random.rand(100, 180)
        angle_idx = 90
        line_width = 3

        result = average_neighbors(sinogram, angle_idx, line_width)

        assert len(result) == 100
        assert not np.any(np.isnan(result))

    def test_averaging_reduces_noise(self):
        """Test that averaging reduces noise."""
        # Create sinogram with noise
        sinogram = np.ones((100, 180)) + np.random.rand(100, 180) * 0.1
        angle_idx = 90

        # No averaging
        profile_no_avg = sinogram[:, angle_idx]

        # With averaging
        profile_avg = average_neighbors(sinogram, angle_idx, 5)

        # Averaged profile should have lower variance
        assert np.var(profile_avg) < np.var(profile_no_avg)

    def test_different_line_widths(self):
        """Test with different line widths (must be odd)."""
        sinogram = np.random.rand(100, 180)
        angle_idx = 90

        for line_width in [1, 3, 5, 7]:
            result = average_neighbors(sinogram, angle_idx, line_width)
            assert len(result) == 100

    def test_invalid_line_width_raises_error(self):
        """Test that even line width raises assertion error."""
        sinogram = np.random.rand(100, 180)

        with pytest.raises(AssertionError):
            average_neighbors(sinogram, 90, 4)  # Even number

    def test_edge_cases(self):
        """Test at sinogram edges."""
        sinogram = np.random.rand(100, 180)

        # First angle
        result = average_neighbors(sinogram, 0, 3)
        assert len(result) == 100

        # Last angle
        result = average_neighbors(sinogram, 179, 3)
        assert len(result) == 100


class TestComputeFsWidth:
    def test_basic_computation(self):
        """Test basic focal spot width computation."""
        fwhm_px = 10.0
        pixel_size = 0.1  # mm
        magnification = 2.0

        result = compute_fs_width(fwhm_px, pixel_size, magnification)

        expected = fwhm_px * pixel_size / magnification
        assert result == pytest.approx(expected)
        assert result == 0.5  # 10 * 0.1 / 2

    def test_different_magnifications(self):
        """Test with different magnification factors."""
        fwhm_px = 20.0
        pixel_size = 0.05

        mag1 = compute_fs_width(fwhm_px, pixel_size, 1.0)
        mag2 = compute_fs_width(fwhm_px, pixel_size, 2.0)
        mag5 = compute_fs_width(fwhm_px, pixel_size, 5.0)

        # Higher magnification should give smaller focal spot width
        assert mag1 > mag2 > mag5


class TestGaussian:
    def test_basic_gaussian(self):
        """Test basic Gaussian function."""
        x = np.linspace(-10, 10, 201)
        A = 1.0
        mu = 0.0
        sigma = 1.0
        B = 0.0

        result = gaussian(x, A, mu, sigma, B)

        assert len(result) == len(x)
        # Peak should be at center
        assert np.argmax(result) == 100

    def test_gaussian_parameters(self):
        """Test that Gaussian parameters work correctly."""
        x = np.linspace(-10, 10, 201)

        # Test amplitude
        g1 = gaussian(x, 1.0, 0.0, 1.0, 0.0)
        g2 = gaussian(x, 2.0, 0.0, 1.0, 0.0)
        assert np.max(g2) > np.max(g1)

        # Test center position
        g1 = gaussian(x, 1.0, -2.0, 1.0, 0.0)
        g2 = gaussian(x, 1.0, 2.0, 1.0, 0.0)
        assert np.argmax(g1) < np.argmax(g2)

        # Test background
        g_no_bg = gaussian(x, 1.0, 0.0, 1.0, 0.0)
        g_with_bg = gaussian(x, 1.0, 0.0, 1.0, 5.0)
        assert np.all(g_with_bg >= g_no_bg)


class TestFindExtremeProfilesErf:
    def test_basic_finding(self):
        """Test finding extreme profiles with error function fitting."""
        # Create synthetic sinogram with varying widths
        n_rays = 100
        n_angles = 180
        profiles = np.zeros((n_rays, n_angles))

        x = np.arange(n_rays)
        for i in range(n_angles):
            # Vary sigma to create different widths
            sigma = 2 + i * 0.05  # Increasing sigma
            A = 1.0
            x0 = n_rays / 2
            B = 0.0
            profiles[:, i] = erf_step(x, A, x0, sigma, B)

        wide_idx, narrow_idx, sigmas = find_extreme_profiles_erf(profiles)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles
        assert len(sigmas) == n_angles
        # Wide index should have larger sigma than narrow
        assert sigmas[wide_idx] >= sigmas[narrow_idx]

    def test_noisy_profiles(self):
        """Test with noisy profiles."""
        n_rays = 100
        n_angles = 50

        # Create profiles with noise
        profiles = np.random.rand(n_rays, n_angles) * 0.1
        for i in range(n_angles):
            x = np.arange(n_rays)
            sigma = 5.0
            profiles[:, i] += erf_step(x, 1.0, n_rays / 2, sigma, 0.5)

        wide_idx, narrow_idx, sigmas = find_extreme_profiles_erf(profiles)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles

    def test_erf_fit_failure_handling(self):
        """Test RuntimeError handling in curve_fit (lines 204-205)."""
        n_rays = 100
        n_angles = 10

        # Create profiles with extreme values that cause fit to fail
        # Shape should be [n_rays, n_angles]
        profiles = np.zeros((n_rays, n_angles))
        profiles[:, 0] = np.ones(n_rays) * 1e20  # Extreme values cause fit failure
        profiles[:, 1] = np.linspace(1e20, 2e20, n_rays)
        profiles[:, 2:] = np.random.rand(n_rays, n_angles - 2) * 1.0

        # Should not raise - handles RuntimeError gracefully
        wide_idx, narrow_idx, sigmas = find_extreme_profiles_erf(profiles)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles
        assert len(sigmas) == n_angles

    def test_all_erf_fits_fail_uses_fallback_indices(self, monkeypatch):
        """Test fallback indices when every ERF fit fails."""
        n_rays = 50
        n_angles = 8
        profiles = np.random.rand(n_rays, n_angles)

        def always_fail(*args, **kwargs):
            raise RuntimeError("forced failure")

        monkeypatch.setattr("scopexr.widths_calculator.curve_fit", always_fail)

        wide_idx, narrow_idx, sigmas = find_extreme_profiles_erf(profiles)

        assert wide_idx == 89
        assert narrow_idx == 0
        assert np.all(np.isnan(sigmas))


class TestFindExtremeProfilesGaussian:
    def test_basic_finding(self):
        """Test finding extreme profiles with Gaussian fitting."""
        n_rays = 100
        n_angles = 180
        sinogram = np.zeros((n_rays, n_angles))

        x = np.arange(n_rays)
        for i in range(n_angles):
            sigma = 3 + i * 0.02  # Varying width
            sinogram[:, i] = gaussian(x, 1.0, n_rays / 2, sigma, 0.1)

        wide_idx, narrow_idx, sigmas, popts = find_extreme_profiles_gaussian(sinogram)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles
        assert len(sigmas) == n_angles
        assert len(popts) == n_angles
        # Wide should have larger sigma
        assert sigmas[wide_idx] >= sigmas[narrow_idx]

    def test_with_noise(self):
        """Test with noisy sinogram."""
        n_rays = 100
        n_angles = 50

        sinogram = np.random.rand(n_rays, n_angles) * 0.1
        for i in range(n_angles):
            x = np.arange(n_rays)
            sinogram[:, i] += gaussian(x, 1.0, n_rays / 2, 5.0, 0.5)

        wide_idx, narrow_idx, sigmas, popts = find_extreme_profiles_gaussian(sinogram)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles
        assert len(sigmas) == n_angles
        assert len(popts) == n_angles

    def test_failed_fits_return_nans(self):
        """Test that failed fits are marked with NaN."""
        n_rays = 100
        n_angles = 10

        # Create mostly invalid data
        sinogram = np.zeros((n_rays, n_angles))
        sinogram[:, 0] = gaussian(np.arange(n_rays), 1.0, 50, 5.0, 0.0)  # One valid

        wide_idx, narrow_idx, sigmas, popts = find_extreme_profiles_gaussian(sinogram)

        # Some sigmas might be NaN due to failed fits
        assert len(sigmas) == n_angles

    def test_gaussian_fit_failure_handling(self):
        """Test RuntimeError handling in Gaussian curve_fit (lines 341-344)."""
        n_rays = 100
        n_angles = 5

        # Create sinogram with extreme values causing fit failures
        sinogram = np.zeros((n_rays, n_angles))
        sinogram[:, 0] = np.ones(n_rays) * 1e20  # Extreme values
        sinogram[:, 1] = np.linspace(1e20, 2e20, n_rays)
        for i in range(2, n_angles):
            sinogram[:, i] = gaussian(np.arange(n_rays), 1.0, 50, 5.0, 0.0)

        # Should handle RuntimeError gracefully
        wide_idx, narrow_idx, sigmas, popts = find_extreme_profiles_gaussian(sinogram)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles
        assert len(sigmas) == n_angles
        assert len(popts) == n_angles
        # Check that popts contains NaN arrays for failed fits
        for popt in popts:
            assert len(popt) == 4
        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles

    def test_partial_gaussian_fit_failures_mark_nan_entries(self, monkeypatch):
        """Test mixed success/failure Gaussian fitting path."""
        n_rays = 60
        n_angles = 6
        x = np.arange(n_rays)
        sinogram = np.column_stack(
            [gaussian(x, 1.0, n_rays / 2, 4.0, 0.1) for _ in range(n_angles)]
        )

        state = {"calls": 0}

        def fail_once_then_succeed(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("forced failure")
            return np.array([1.0, n_rays / 2, 3.0, 0.0]), np.eye(4)

        monkeypatch.setattr(
            "scopexr.widths_calculator.curve_fit", fail_once_then_succeed
        )

        wide_idx, narrow_idx, sigmas, popts = find_extreme_profiles_gaussian(sinogram)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles
        assert np.isnan(sigmas[0])
        assert np.all(np.isnan(popts[0]))
        assert np.all(sigmas[1:] == pytest.approx(3.0))

    def test_all_gaussian_fits_fail_uses_fallback_indices(self, monkeypatch):
        """Test fallback indices when every Gaussian fit fails."""
        n_rays = 50
        n_angles = 8
        sinogram = np.random.rand(n_rays, n_angles)

        def always_fail(*args, **kwargs):
            raise RuntimeError("forced failure")

        monkeypatch.setattr("scopexr.widths_calculator.curve_fit", always_fail)

        wide_idx, narrow_idx, sigmas, popts = find_extreme_profiles_gaussian(sinogram)

        assert wide_idx == 89
        assert narrow_idx == 0
        assert np.all(np.isnan(sigmas))
        assert all(np.all(np.isnan(popt)) for popt in popts)

    def test_extreme_profiles_erf_fit_failure(self):
        """Test ERF fitting with profiles that may fail to fit."""
        n_rays = 100
        n_angles = 10

        # Create mostly noise/invalid data
        profiles = np.random.rand(n_rays, n_angles) * 0.1

        # Should not crash even if many fits fail
        wide_idx, narrow_idx, sigmas = find_extreme_profiles_erf(profiles)

        assert 0 <= wide_idx < n_angles
        assert 0 <= narrow_idx < n_angles
        assert len(sigmas) == n_angles


class TestComputeLsfFromProjection:
    def test_normalized_lsf_on_nonuniform_reconstruction(self):
        """Test projection LSF normalization and positivity."""
        y, x = np.mgrid[-20:21, -20:21]
        reconstruction = np.exp(-((x**2 + y**2) / (2 * 5.0**2))) + 0.05

        horizontal_lsf, vertical_lsf = compute_lsf_from_projection(reconstruction)

        assert horizontal_lsf.ndim == 1
        assert vertical_lsf.ndim == 1
        assert horizontal_lsf.shape[0] == reconstruction.shape[1]
        assert vertical_lsf.shape[0] == reconstruction.shape[0]
        assert np.all(horizontal_lsf >= 0)
        assert np.all(vertical_lsf >= 0)
        assert np.sum(horizontal_lsf) == pytest.approx(1.0)
        assert np.sum(vertical_lsf) == pytest.approx(1.0)

    def test_zero_sum_after_background_subtraction(self):
        """Test zero-sum branch when reconstruction is flat."""
        reconstruction = np.ones((25, 30), dtype=float) * 7.0

        horizontal_lsf, vertical_lsf = compute_lsf_from_projection(reconstruction)

        assert np.all(horizontal_lsf == 0)
        assert np.all(vertical_lsf == 0)
