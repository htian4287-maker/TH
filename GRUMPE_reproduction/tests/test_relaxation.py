from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from grumpe_method.photoclinometry import reference_slopes
from grumpe_method.reflectance import HapkeParameters
from grumpe_method.relaxation import gaussian_kernel, solve_paper_relaxation
from grumpe_method.synthetic import make_synthetic_case


def test_gaussian_kernel_is_normalized_and_symmetric() -> None:
    kernel = gaussian_kernel(3.0)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1.0e-14)


def test_separable_gaussian_matches_dense_kernel() -> None:
    rng = np.random.default_rng(8)
    values = rng.normal(size=(20, 22))
    kernel = gaussian_kernel(3.0)
    dense = convolve(values, kernel, mode="reflect")
    separable = gaussian_filter(values, 3.0, mode="reflect", truncate=4.0)
    np.testing.assert_allclose(separable, dense, atol=2.0e-15, rtol=2.0e-15)


def test_published_relaxation_reduces_objective_and_dem_error() -> None:
    case = make_synthetic_case(size=32, noise_std=0.0, parameters=HapkeParameters())
    p, q = reference_slopes(case.truth_dem, case.pixel_size_m, case.pixel_size_m)
    result = solve_paper_relaxation(
        p + 0.015,
        q - 0.012,
        case.initial_dem,
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        depth_weight=0.2,
        lowpass_sigma_px=4.0,
        max_successful_iterations=300,
        max_steps_without_improvement=40,
        max_total_updates=400,
    )
    initial_error = result.history[0]["error"]
    initial_rmse = np.sqrt(np.mean((case.initial_dem - case.truth_dem) ** 2))
    recovered_rmse = np.sqrt(np.mean((result.dem - case.truth_dem) ** 2))
    assert np.isfinite(result.dem).all()
    assert result.best_error < initial_error
    assert recovered_rmse < initial_rmse
