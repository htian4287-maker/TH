from __future__ import annotations

import numpy as np

from grumpe_method.photoclinometry import estimate_phcl, reference_slopes
from grumpe_method.reflectance import HapkeParameters
from grumpe_method.synthetic import make_synthetic_case


def test_multimage_phcl_improves_prior_slope_rmse() -> None:
    parameters = HapkeParameters()
    case = make_synthetic_case(size=36, noise_std=2.0e-5, parameters=parameters)
    p0, q0 = reference_slopes(case.initial_dem, case.pixel_size_m, case.pixel_size_m)
    before = np.sqrt(np.mean((p0 - case.truth_p) ** 2 + (q0 - case.truth_q) ** 2))
    result = estimate_phcl(
        case.images,
        case.sun_directions,
        case.view_directions,
        case.initial_dem,
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        parameters=parameters,
        dem_weight=2.0e-4,
        dem_sigma_px=5.0,
        max_iterations=35,
    )
    after = np.sqrt(
        np.mean(
            ((result.p - case.truth_p) ** 2 + (result.q - case.truth_q) ** 2)[
                result.valid_mask
            ]
        )
    )
    assert after < before


def test_phcl_accepts_explicit_reference_slopes() -> None:
    parameters = HapkeParameters()
    case = make_synthetic_case(size=32, noise_std=2.0e-5, parameters=parameters)
    p0, q0 = reference_slopes(case.initial_dem, case.pixel_size_m, case.pixel_size_m)
    result = estimate_phcl(
        case.images,
        case.sun_directions,
        case.view_directions,
        case.initial_dem,
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        parameters=parameters,
        reference_p=p0,
        reference_q=q0,
        max_iterations=2,
    )
    assert result.p.shape == case.initial_dem.shape
    assert result.q.shape == case.initial_dem.shape


def test_phcl_can_warm_start_optimizer_state() -> None:
    parameters = HapkeParameters()
    case = make_synthetic_case(size=32, noise_std=2.0e-5, parameters=parameters)
    first = estimate_phcl(
        case.images,
        case.sun_directions,
        case.view_directions,
        case.initial_dem,
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        parameters=parameters,
        max_iterations=2,
    )
    continued = estimate_phcl(
        case.images,
        case.sun_directions,
        case.view_directions,
        case.initial_dem,
        pixel_size_x=case.pixel_size_m,
        pixel_size_y=case.pixel_size_m,
        parameters=parameters,
        warm_start_p=first.p,
        warm_start_q=first.q,
        warm_start_albedo=first.single_scattering_albedo,
        initial_damping=float(first.history[-1]["damping"]),
        previous_error=float(first.history[-1]["total_error"]),
        iteration_offset=2,
        max_iterations=2,
    )
    assert continued.history[0]["iteration"] == 3.0
    assert np.isfinite(continued.p).all()
    assert np.isfinite(continued.q).all()
