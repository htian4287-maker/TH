"""Hapke reflectance components used in Grumpe et al. (2014).

The Grumpe system compares Hapke's isotropic (IMSA) and anisotropic
multiple-scattering approximations (AMSA). The common reflectance follows
Eq. (17)--(20) of the Advances in Space Research paper. AMSA's Legendre
expansion follows Hapke (2002), while the optional macroscopic-roughness
correction follows Hapke (1984). Values are bidirectional reflectance
(sr^-1), not image DN and not radiance factor I/F.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss, legval


PhaseFunction = Literal["dhg", "cornette_shanks"]
ReflectanceModel = Literal["imsa", "amsa"]
HFunctionApproximation = Literal["simple", "level2"]


@dataclass(frozen=True)
class HapkeParameters:
    """Fixed or spatially varying Hapke parameters for one wavelength."""

    single_scattering_albedo: float | np.ndarray = 0.4
    opposition_amplitude: float = 1.0
    opposition_width: float = 0.06
    phase_function: PhaseFunction = "dhg"
    dhg_b: float = 0.30
    dhg_c: float = 0.70
    cornette_shanks_n: float = 0.20
    roughness_degrees: float = 0.0
    h_function: HFunctionApproximation = "simple"
    legendre_order: int = 15


def warell_moon_2004(
    single_scattering_albedo: float | np.ndarray = 0.4,
    *,
    h_function: HFunctionApproximation = "level2",
    legendre_order: int = 15,
) -> HapkeParameters:
    """Return the fixed lunar Hapke parameters adopted by Grumpe et al.

    Warell's lunar solution uses b=0.21, c=0.70, mean roughness=11 degrees,
    B_S0=3.1 and h=0.11.  Only the single-scattering albedo remains variable.
    """
    return HapkeParameters(
        single_scattering_albedo=single_scattering_albedo,
        opposition_amplitude=3.1,
        opposition_width=0.11,
        phase_function="dhg",
        dhg_b=0.21,
        dhg_c=0.70,
        roughness_degrees=11.0,
        h_function=h_function,
        legendre_order=legendre_order,
    )


def shadow_hiding_opposition(
    phase_radians: np.ndarray,
    amplitude: float,
    width: float,
) -> np.ndarray:
    """Hapke shadow-hiding term B_SH(alpha), Grumpe paper Eq. (18)."""
    if amplitude < 0.0 or width <= 0.0:
        raise ValueError("Opposition amplitude must be non-negative and width positive")
    alpha = np.asarray(phase_radians, dtype=np.float64)
    return 1.0 + amplitude / (1.0 + np.tan(0.5 * alpha) / width)


def phase_double_henyey_greenstein(
    phase_radians: np.ndarray,
    b: float,
    c: float,
) -> np.ndarray:
    """Two-parameter double Henyey--Greenstein function, paper Eq. (19)."""
    if not 0.0 <= b < 1.0:
        raise ValueError("DHG lobe width b must lie in [0, 1)")
    if not -1.0 <= c <= 1.0:
        raise ValueError("DHG mixing c must lie in [-1, 1]")
    cosine = np.cos(np.asarray(phase_radians, dtype=np.float64))
    scale = 1.0 - b * b
    backward = scale / np.maximum(1.0 + 2.0 * b * cosine + b * b, 1.0e-15) ** 1.5
    forward = scale / np.maximum(1.0 - 2.0 * b * cosine + b * b, 1.0e-15) ** 1.5
    return 0.5 * (1.0 + c) * backward + 0.5 * (1.0 - c) * forward


def phase_cornette_shanks(phase_radians: np.ndarray, n: float) -> np.ndarray:
    """One-parameter Cornette--Shanks phase function, paper Eq. (20)."""
    if not -0.999 < n < 0.999:
        raise ValueError("Cornette--Shanks asymmetry n must lie in (-0.999, 0.999)")
    cosine = np.cos(np.asarray(phase_radians, dtype=np.float64))
    numerator = 1.5 * (1.0 - n * n) * (1.0 + cosine * cosine)
    denominator = (2.0 + n * n) * np.maximum(
        1.0 + n * n - 2.0 * n * cosine, 1.0e-15
    ) ** 1.5
    return numerator / denominator


def particle_phase(phase_radians: np.ndarray, parameters: HapkeParameters) -> np.ndarray:
    """Evaluate the selected single-particle phase function."""
    if parameters.phase_function == "dhg":
        return phase_double_henyey_greenstein(
            phase_radians, parameters.dhg_b, parameters.dhg_c
        )
    if parameters.phase_function == "cornette_shanks":
        return phase_cornette_shanks(
            phase_radians, parameters.cornette_shanks_n
        )
    raise ValueError(f"Unsupported phase function: {parameters.phase_function}")


def chandrashekhar_h_imsa(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Hapke's simple IMSA approximation to the Chandrasekhar H-function."""
    mu_values = np.asarray(mu, dtype=np.float64)
    albedo = np.asarray(w, dtype=np.float64)
    gamma = np.sqrt(np.maximum(1.0 - albedo, 0.0))
    return (1.0 + 2.0 * mu_values) / np.maximum(
        1.0 + 2.0 * gamma * mu_values, 1.0e-15
    )


def chandrashekhar_h_level2(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Hapke (2002) level-2 approximation to the H-function, Eqs. (12--13)."""
    x = np.maximum(np.asarray(mu, dtype=np.float64), 1.0e-12)
    albedo = np.asarray(w, dtype=np.float64)
    gamma = np.sqrt(np.maximum(1.0 - albedo, 0.0))
    r0 = (1.0 - gamma) / np.maximum(1.0 + gamma, 1.0e-15)
    logarithm = np.log1p(x) - np.log(x)
    bracket = r0 + 0.5 * (1.0 - 2.0 * r0 * x) * logarithm
    return 1.0 / np.maximum(1.0 - albedo * x * bracket, 1.0e-15)


def _h_function(
    mu: np.ndarray,
    w: np.ndarray,
    approximation: HFunctionApproximation,
) -> np.ndarray:
    if approximation == "simple":
        return chandrashekhar_h_imsa(mu, w)
    if approximation == "level2":
        return chandrashekhar_h_level2(mu, w)
    raise ValueError(f"Unsupported H-function approximation: {approximation}")


def hapke_a_coefficients(order: int = 15) -> np.ndarray:
    """Return Hapke (2002) coefficients a_n used by AMSA."""
    if order < 1:
        raise ValueError("Legendre order must be at least one")
    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[1] = -0.5
    for degree in range(3, order + 1, 2):
        coefficients[degree] = (
            (2.0 - degree) / (degree + 1.0) * coefficients[degree - 2]
        )
    return coefficients


@lru_cache(maxsize=64)
def _phase_legendre_coefficients_cached(
    phase_function: PhaseFunction,
    dhg_b: float,
    dhg_c: float,
    cornette_shanks_n: float,
    order: int,
) -> tuple[float, ...]:
    if order < 1:
        raise ValueError("Legendre order must be at least one")
    degrees = np.arange(order + 1)
    if phase_function == "dhg":
        if not 0.0 <= dhg_b < 1.0 or not -1.0 <= dhg_c <= 1.0:
            raise ValueError("Invalid DHG parameters")
        coefficients = (2.0 * degrees + 1.0) * dhg_b**degrees
        # Eq. (19) uses the +2b cos(alpha) lobe for (1+c)/2.
        coefficients[1::2] *= -dhg_c
    elif phase_function == "cornette_shanks":
        # Numerical projection avoids a fragile closed-form expansion and is
        # deterministic to substantially better precision than the inversion.
        nodes, weights = leggauss(max(96, 4 * order + 16))
        phase_values = phase_cornette_shanks(
            np.arccos(np.clip(nodes, -1.0, 1.0)), cornette_shanks_n
        )
        coefficients = np.empty(order + 1, dtype=np.float64)
        for degree in range(order + 1):
            basis = np.zeros(degree + 1)
            basis[-1] = 1.0
            coefficients[degree] = (
                0.5
                * (2.0 * degree + 1.0)
                * np.sum(weights * phase_values * legval(nodes, basis))
            )
    else:
        raise ValueError(f"Unsupported phase function: {phase_function}")
    return tuple(float(value) for value in coefficients)


def phase_legendre_coefficients(parameters: HapkeParameters) -> np.ndarray:
    """Legendre coefficients b_n of the selected particle phase function."""
    return np.asarray(
        _phase_legendre_coefficients_cached(
            parameters.phase_function,
            float(parameters.dhg_b),
            float(parameters.dhg_c),
            float(parameters.cornette_shanks_n),
            int(parameters.legendre_order),
        ),
        dtype=np.float64,
    )


def amsa_directional_p(
    cosine: np.ndarray,
    parameters: HapkeParameters,
) -> np.ndarray:
    """Evaluate Hapke's directional P(mu) term for AMSA."""
    b_n = phase_legendre_coefficients(parameters)
    a_n = hapke_a_coefficients(parameters.legendre_order)
    return 1.0 + legval(np.clip(cosine, -1.0, 1.0), a_n * b_n)


def amsa_mean_p(parameters: HapkeParameters) -> float:
    """Evaluate the scalar mean-P term in Hapke's AMSA expansion."""
    b_n = phase_legendre_coefficients(parameters)
    a_n = hapke_a_coefficients(parameters.legendre_order)
    return float(1.0 + np.sum(a_n * a_n * b_n))


def anisotropic_multiple_scattering(
    mu0: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    parameters: HapkeParameters,
) -> np.ndarray:
    """Hapke (2002) AMSA multiple-scattering term M(mu0, mu)."""
    h0_minus_one = chandrashekhar_h_level2(mu0, w) - 1.0
    h_minus_one = chandrashekhar_h_level2(mu, w) - 1.0
    p0 = amsa_directional_p(mu0, parameters)
    p = amsa_directional_p(mu, parameters)
    mean_p = amsa_mean_p(parameters)
    return (
        p0 * h_minus_one
        + p * h0_minus_one
        + mean_p * h0_minus_one * h_minus_one
    )


def macroscopic_roughness_correction(
    mu0: np.ndarray,
    mu: np.ndarray,
    phase_radians: np.ndarray,
    roughness_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Hapke (1984) roughness factor S and effective mu0/mu."""
    if not 0.0 <= roughness_degrees < 90.0:
        raise ValueError("Macroscopic roughness must lie in [0, 90) degrees")
    mu0_values, mu_values, phase = np.broadcast_arrays(
        np.asarray(mu0, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
        np.asarray(phase_radians, dtype=np.float64),
    )
    if roughness_degrees == 0.0:
        return np.ones_like(mu0_values), mu0_values.copy(), mu_values.copy()

    mu0_clip = np.clip(mu0_values, 1.0e-12, 1.0)
    mu_clip = np.clip(mu_values, 1.0e-12, 1.0)
    incidence = np.arccos(mu0_clip)
    emission = np.arccos(mu_clip)
    sin_i = np.sin(incidence)
    sin_e = np.sin(emission)
    azimuth_denominator = sin_i * sin_e
    cos_azimuth = np.where(
        azimuth_denominator > 1.0e-12,
        (np.cos(phase) - mu0_clip * mu_clip)
        / np.maximum(azimuth_denominator, 1.0e-12),
        1.0,
    )
    azimuth = np.arccos(np.clip(cos_azimuth, -1.0, 1.0))

    theta = np.deg2rad(roughness_degrees)
    tan_theta = np.tan(theta)
    cot_theta = 1.0 / tan_theta
    chi = 1.0 / np.sqrt(1.0 + np.pi * tan_theta * tan_theta)

    def e_terms(angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cot_angle = np.cos(angle) / np.maximum(np.sin(angle), 1.0e-15)
        e1 = np.exp(-(2.0 / np.pi) * cot_theta * cot_angle)
        e2 = np.exp(-(1.0 / np.pi) * cot_theta**2 * cot_angle**2)
        return e1, e2

    e1_i, e2_i = e_terms(incidence)
    e1_e, e2_e = e_terms(emission)
    eta_i = chi * (
        mu0_clip + sin_i * tan_theta * e2_i / np.maximum(2.0 - e1_i, 1.0e-15)
    )
    eta_e = chi * (
        mu_clip + sin_e * tan_theta * e2_e / np.maximum(2.0 - e1_e, 1.0e-15)
    )
    sin_half_squared = np.sin(0.5 * azimuth) ** 2

    incidence_le_emission = incidence <= emission
    denominator_ie = np.maximum(
        2.0 - e1_e - (azimuth / np.pi) * e1_i, 1.0e-15
    )
    mu0_ie = chi * (
        mu0_clip
        + sin_i
        * tan_theta
        * (np.cos(azimuth) * e2_e + sin_half_squared * e2_i)
        / denominator_ie
    )
    mu_ie = chi * (
        mu_clip
        + sin_e
        * tan_theta
        * (e2_e - sin_half_squared * e2_i)
        / denominator_ie
    )

    denominator_ei = np.maximum(
        2.0 - e1_i - (azimuth / np.pi) * e1_e, 1.0e-15
    )
    mu0_ei = chi * (
        mu0_clip
        + sin_i
        * tan_theta
        * (e2_i - sin_half_squared * e2_e)
        / denominator_ei
    )
    mu_ei = chi * (
        mu_clip
        + sin_e
        * tan_theta
        * (np.cos(azimuth) * e2_i + sin_half_squared * e2_e)
        / denominator_ei
    )
    mu0_effective = np.where(incidence_le_emission, mu0_ie, mu0_ei)
    mu_effective = np.where(incidence_le_emission, mu_ie, mu_ei)

    overlap = np.exp(-2.0 * np.tan(0.5 * azimuth))
    common = (
        mu_effective
        / np.maximum(eta_e, 1.0e-15)
        * mu0_clip
        / np.maximum(eta_i, 1.0e-15)
        * chi
    )
    denominator_s_ie = 1.0 - overlap + overlap * chi * (
        mu0_clip / np.maximum(eta_i, 1.0e-15)
    )
    denominator_s_ei = 1.0 - overlap + overlap * chi * (
        mu_clip / np.maximum(eta_e, 1.0e-15)
    )
    shadow = common / np.maximum(
        np.where(incidence_le_emission, denominator_s_ie, denominator_s_ei),
        1.0e-15,
    )
    return (
        np.maximum(shadow, 0.0),
        np.clip(mu0_effective, 1.0e-12, 1.0),
        np.clip(mu_effective, 1.0e-12, 1.0),
    )


def hapke_reflectance(
    mu0: np.ndarray,
    mu: np.ndarray,
    phase_radians: np.ndarray,
    parameters: HapkeParameters,
    *,
    multiple_scattering: ReflectanceModel,
) -> np.ndarray:
    """Evaluate IMSA or AMSA bidirectional reflectance with shared settings."""
    mu0_values, mu_values, phase = np.broadcast_arrays(
        np.asarray(mu0, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
        np.asarray(phase_radians, dtype=np.float64),
    )
    w = np.asarray(parameters.single_scattering_albedo, dtype=np.float64)
    if np.any((w <= 0.0) | (w >= 1.0)):
        raise ValueError("Single-scattering albedo must lie strictly between 0 and 1")
    shadow, mu0_effective, mu_effective = macroscopic_roughness_correction(
        mu0_values, mu_values, phase, parameters.roughness_degrees
    )
    if multiple_scattering == "imsa":
        h0 = _h_function(mu0_effective, w, parameters.h_function)
        h = _h_function(mu_effective, w, parameters.h_function)
        multiple = h0 * h - 1.0
    elif multiple_scattering == "amsa":
        multiple = anisotropic_multiple_scattering(
            mu0_effective, mu_effective, w, parameters
        )
    else:
        raise ValueError(f"Unsupported multiple-scattering model: {multiple_scattering}")
    single = particle_phase(phase, parameters) * shadow_hiding_opposition(
        phase, parameters.opposition_amplitude, parameters.opposition_width
    )
    reflectance = (
        w
        / (4.0 * np.pi)
        * mu0_effective
        / np.maximum(mu0_effective + mu_effective, 1.0e-15)
        * (single + multiple)
        * shadow
    )
    valid = (mu0_values > 0.0) & (mu_values > 0.0)
    return np.where(valid, reflectance, np.nan)


def hapke_imsa(
    mu0: np.ndarray,
    mu: np.ndarray,
    phase_radians: np.ndarray,
    parameters: HapkeParameters,
) -> np.ndarray:
    """Evaluate Hapke IMSA bidirectional reflectance."""
    return hapke_reflectance(
        mu0, mu, phase_radians, parameters, multiple_scattering="imsa"
    )


def hapke_amsa(
    mu0: np.ndarray,
    mu: np.ndarray,
    phase_radians: np.ndarray,
    parameters: HapkeParameters,
) -> np.ndarray:
    """Evaluate Hapke (2002) AMSA bidirectional reflectance."""
    return hapke_reflectance(
        mu0, mu, phase_radians, parameters, multiple_scattering="amsa"
    )
