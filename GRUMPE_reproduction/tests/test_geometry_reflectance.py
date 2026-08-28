from __future__ import annotations

import numpy as np

from grumpe_method.geometry import (
    direction_from_azimuth_zenith,
    normals_to_slopes,
    slopes_to_normals,
)
from grumpe_method.reflectance import (
    HapkeParameters,
    hapke_amsa,
    hapke_imsa,
    macroscopic_roughness_correction,
    phase_legendre_coefficients,
    phase_cornette_shanks,
    phase_double_henyey_greenstein,
)


def test_slopes_normals_round_trip() -> None:
    p = np.array([[0.1, -0.2], [0.3, 0.0]])
    q = np.array([[-0.1, 0.25], [0.0, -0.4]])
    p2, q2 = normals_to_slopes(slopes_to_normals(p, q))
    np.testing.assert_allclose(p2, p, atol=1.0e-12)
    np.testing.assert_allclose(q2, q, atol=1.0e-12)


def test_direction_convention() -> None:
    east_horizon = direction_from_azimuth_zenith(90.0, 90.0)
    np.testing.assert_allclose(east_horizon, [1.0, 0.0, 0.0], atol=1.0e-12)


def test_phase_functions_are_positive() -> None:
    phase = np.linspace(0.0, np.pi, 200)
    assert np.all(phase_double_henyey_greenstein(phase, 0.3, 0.7) > 0.0)
    assert np.all(phase_cornette_shanks(phase, 0.2) > 0.0)


def test_hapke_reflectance_increases_with_albedo() -> None:
    mu0 = np.array([0.4, 0.7])
    mu = np.array([0.8, 0.9])
    phase = np.deg2rad([40.0, 55.0])
    low = hapke_imsa(mu0, mu, phase, HapkeParameters(single_scattering_albedo=0.25))
    high = hapke_imsa(mu0, mu, phase, HapkeParameters(single_scattering_albedo=0.65))
    assert np.all(np.isfinite(low))
    assert np.all(high > low)


def test_dhg_legendre_series_matches_phase_function() -> None:
    parameters = HapkeParameters(dhg_b=0.3, dhg_c=0.7, legendre_order=30)
    cosine = np.linspace(-1.0, 1.0, 301)
    reconstructed = np.polynomial.legendre.legval(
        cosine, phase_legendre_coefficients(parameters)
    )
    direct = phase_double_henyey_greenstein(
        np.arccos(cosine), parameters.dhg_b, parameters.dhg_c
    )
    np.testing.assert_allclose(reconstructed, direct, atol=2.0e-13, rtol=2.0e-13)


def test_amsa_reduces_to_level2_imsa_for_isotropic_particles() -> None:
    parameters = HapkeParameters(
        dhg_b=0.0,
        dhg_c=0.0,
        h_function="level2",
        legendre_order=15,
    )
    mu0 = np.array([0.35, 0.62, 0.85])
    mu = np.array([0.92, 0.80, 0.67])
    phase = np.deg2rad([35.0, 52.0, 68.0])
    np.testing.assert_allclose(
        hapke_amsa(mu0, mu, phase, parameters),
        hapke_imsa(mu0, mu, phase, parameters),
        atol=1.0e-14,
        rtol=1.0e-13,
    )


def test_macroscopic_roughness_zero_is_identity_and_eleven_degrees_is_finite() -> None:
    mu0 = np.cos(np.deg2rad([35.0, 55.0, 65.0]))
    mu = np.cos(np.deg2rad([5.0, 8.0, 12.0]))
    phase = np.deg2rad([32.0, 51.0, 59.0])
    shadow0, mu00, mu_zero = macroscopic_roughness_correction(mu0, mu, phase, 0.0)
    np.testing.assert_allclose(shadow0, 1.0)
    np.testing.assert_allclose(mu00, mu0)
    np.testing.assert_allclose(mu_zero, mu)
    shadow, mu0e, mue = macroscopic_roughness_correction(mu0, mu, phase, 11.0)
    assert np.all(np.isfinite(shadow))
    assert np.all(shadow > 0.0)
    assert np.all((mu0e > 0.0) & (mu0e <= 1.0))
    assert np.all((mue > 0.0) & (mue <= 1.0))


def test_amsa_and_imsa_differ_for_anisotropic_particles() -> None:
    parameters = HapkeParameters(
        single_scattering_albedo=0.55,
        dhg_b=0.3,
        dhg_c=0.7,
        h_function="level2",
        roughness_degrees=11.0,
    )
    mu0 = np.cos(np.deg2rad([35.0, 55.0, 65.0]))
    mu = np.cos(np.deg2rad([5.0, 8.0, 12.0]))
    phase = np.deg2rad([32.0, 51.0, 59.0])
    imsa = hapke_imsa(mu0, mu, phase, parameters)
    amsa = hapke_amsa(mu0, mu, phase, parameters)
    assert np.max(np.abs(imsa - amsa)) > 1.0e-5
