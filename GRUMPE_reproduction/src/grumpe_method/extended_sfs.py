"""Extended multi-image shape-from-shading after PHCL initialization.

The solver follows Grumpe et al. (2014), Eqs. (42)--(47): reflectance error,
integrability between the current DEM and p/q, and a Gaussian low-pass DEM
gradient constraint are alternately minimized.  Per-pixel 2x2 Gauss--Newton
blocks update p/q; Eq. (47) is applied as damped Poisson relaxation for z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .photoclinometry import _lowpass_normal_diagonal, _update_albedo, render_hapke_stack
from .reflectance import HapkeParameters, ReflectanceModel


@dataclass(frozen=True)
class ExtendedSFSResult:
    dem: np.ndarray
    p: np.ndarray
    q: np.ndarray
    single_scattering_albedo: np.ndarray
    modeled_images: np.ndarray
    valid_mask: np.ndarray
    history: tuple[dict[str, float], ...]
    converged: bool


def slopes_from_dem(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    row_slope, p = np.gradient(np.asarray(dem, dtype=np.float64), dy, dx)
    return p, -row_slope


def _lowpass(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return values
    return gaussian_filter(values, sigma, mode="reflect", truncate=4.0)


def _poisson_relaxation(
    z: np.ndarray,
    p: np.ndarray,
    q_north: np.ndarray,
    dx: float,
    dy: float,
    sweeps: int = 12,
) -> np.ndarray:
    """Damped Eq. (47) updates with Neumann-like reflected boundaries."""
    current = np.asarray(z, dtype=np.float64).copy()
    dpdx = np.gradient(p, dx, axis=1)
    dqdy = -np.gradient(q_north, dy, axis=0)
    divergence = dpdx + dqdy
    denominator = 2.0 / dx**2 + 2.0 / dy**2
    target_mean = float(np.mean(current))
    for _ in range(sweeps):
        padded = np.pad(current, 1, mode="edge")
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        candidate = (
            (left + right) / dx**2
            + (up + down) / dy**2
            - divergence
        ) / denominator
        current = 0.55 * candidate + 0.45 * current
        current += target_mean - float(np.mean(current))
    return current


def _objective(
    model: np.ndarray,
    observed: np.ndarray,
    usable: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    *,
    dx: float,
    dy: float,
    integrability_weight: float,
    dem_weight: float,
    dem_sigma_px: float,
) -> tuple[float, float, float, float]:
    active = usable & np.isfinite(model) & np.isfinite(observed)
    residual = np.where(active, model - observed, 0.0)
    data_error = 0.5 * float(np.sum(residual * residual))
    pz, qz = slopes_from_dem(z, dx, dy)
    integrability = 0.5 * integrability_weight * float(
        np.sum((p - pz) ** 2 + (q - qz) ** 2)
    )
    prior_p = _lowpass(p - p0, dem_sigma_px)
    prior_q = _lowpass(q - q0, dem_sigma_px)
    dem_error = 0.5 * dem_weight * float(np.sum(prior_p**2 + prior_q**2))
    return data_error + integrability + dem_error, data_error, integrability, dem_error


def estimate_extended_sfs(
    observed_images: np.ndarray,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    reference_dem: np.ndarray,
    *,
    initial_dem: np.ndarray,
    initial_p: np.ndarray,
    initial_q: np.ndarray,
    initial_albedo: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    parameters: HapkeParameters,
    reflectance_model: ReflectanceModel = "amsa",
    integrability_weight: float = 2.5e-4,
    dem_weight: float = 2.0e-4,
    dem_sigma_px: float = 15.0,
    albedo_sigma_px: float = 21.0,
    max_iterations: int = 40,
    relative_tolerance: float = 1.0e-6,
    iteration_offset: int = 0,
) -> ExtendedSFSResult:
    observed = np.asarray(observed_images, dtype=np.float64)
    sun = np.asarray(sun_directions, dtype=np.float64)
    view = np.asarray(view_directions, dtype=np.float64)
    reference = np.asarray(reference_dem, dtype=np.float64)
    z = np.asarray(initial_dem, dtype=np.float64).copy()
    p = np.asarray(initial_p, dtype=np.float64).copy()
    q = np.asarray(initial_q, dtype=np.float64).copy()
    w = np.clip(np.asarray(initial_albedo, dtype=np.float64).copy(), 0.02, 0.98)
    if observed.ndim != 3 or observed.shape[1:] != reference.shape:
        raise ValueError("Images and DEM must share one grid")
    if min(reference.shape) < 5:
        raise ValueError("Extended SfS needs a grid of at least 5x5")
    p0, q0 = slopes_from_dem(reference, pixel_size_x, pixel_size_y)
    base = render_hapke_stack(p0, q0, np.full(reference.shape, 0.4), sun, view, parameters, reflectance_model)
    usable = np.isfinite(observed) & np.isfinite(base) & (observed > 0.0)
    valid_mask = usable.sum(axis=0) >= min(3, observed.shape[0])
    usable &= valid_mask[None]
    if not valid_mask.any():
        raise ValueError("No valid multi-image pixels for extended SfS")

    damping = 1.0e-4
    diagonal_prior = dem_weight * _lowpass_normal_diagonal(dem_sigma_px)
    history: list[dict[str, float]] = []
    previous = np.inf
    converged = False
    slope_epsilon = 1.0e-4
    rejected = 0

    for iteration in range(max_iterations):
        w = _update_albedo(
            observed, usable, p, q, w, sun, view, parameters,
            reflectance_model, sigma_px=albedo_sigma_px, iterations=3,
        )
        model = render_hapke_stack(p, q, w, sun, view, parameters, reflectance_model)
        current = _objective(
            model, observed, usable, z, p, q, p0, q0,
            dx=pixel_size_x, dy=pixel_size_y,
            integrability_weight=integrability_weight,
            dem_weight=dem_weight, dem_sigma_px=dem_sigma_px,
        )
        jp = (
            render_hapke_stack(p + slope_epsilon, q, w, sun, view, parameters, reflectance_model)
            - render_hapke_stack(p - slope_epsilon, q, w, sun, view, parameters, reflectance_model)
        ) / (2.0 * slope_epsilon)
        jq = (
            render_hapke_stack(p, q + slope_epsilon, w, sun, view, parameters, reflectance_model)
            - render_hapke_stack(p, q - slope_epsilon, w, sun, view, parameters, reflectance_model)
        ) / (2.0 * slope_epsilon)
        active = usable & np.isfinite(model) & np.isfinite(jp) & np.isfinite(jq)
        residual = np.where(active, model - observed, 0.0)
        jpa = np.where(active, jp, 0.0)
        jqa = np.where(active, jq, 0.0)
        pz, qz = slopes_from_dem(z, pixel_size_x, pixel_size_y)
        gp = np.sum(jpa * residual, axis=0)
        gq = np.sum(jqa * residual, axis=0)
        gp += integrability_weight * (p - pz)
        gq += integrability_weight * (q - qz)
        gp += dem_weight * _lowpass(_lowpass(p - p0, dem_sigma_px), dem_sigma_px)
        gq += dem_weight * _lowpass(_lowpass(q - q0, dem_sigma_px), dem_sigma_px)
        hpp = np.sum(jpa * jpa, axis=0) + integrability_weight + diagonal_prior + damping
        hqq = np.sum(jqa * jqa, axis=0) + integrability_weight + diagonal_prior + damping
        hpq = np.sum(jpa * jqa, axis=0)
        determinant = np.maximum(hpp * hqq - hpq * hpq, 1.0e-15)
        dp = np.clip((-hqq * gp + hpq * gq) / determinant, -0.15, 0.15)
        dq = np.clip((hpq * gp - hpp * gq) / determinant, -0.15, 0.15)

        accepted = False
        best = current
        best_state = (z, p, q, model)
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            pc = np.where(valid_mask, np.clip(p + alpha * dp, -3.0, 3.0), p0)
            qc = np.where(valid_mask, np.clip(q + alpha * dq, -3.0, 3.0), q0)
            z_relaxed = _poisson_relaxation(z, pc, qc, pixel_size_x, pixel_size_y)
            zc = z + alpha * (z_relaxed - z)
            zc += float(np.mean(reference - zc))
            mc = render_hapke_stack(pc, qc, w, sun, view, parameters, reflectance_model)
            score = _objective(
                mc, observed, usable, zc, pc, qc, p0, q0,
                dx=pixel_size_x, dy=pixel_size_y,
                integrability_weight=integrability_weight,
                dem_weight=dem_weight, dem_sigma_px=dem_sigma_px,
            )
            if score[0] < best[0]:
                accepted = True
                best = score
                best_state = (zc, pc, qc, mc)
                break
        if accepted:
            z, p, q, model = best_state
            damping = max(damping * 0.2, 1.0e-10)
            rejected = 0
        else:
            damping = min(damping * 1.5, 1.0e8)
            rejected += 1
        relative = abs(previous - best[0]) / max(abs(previous), 1.0) if np.isfinite(previous) else np.inf
        history.append({
            "iteration": float(iteration_offset + iteration + 1),
            "total_error": float(best[0]),
            "data_error": float(best[1]),
            "integrability_error": float(best[2]),
            "dem_error": float(best[3]),
            "albedo_sigma_px": float(albedo_sigma_px),
            "accepted": float(accepted),
            "damping": float(damping),
            "relative_change": float(relative),
        })
        if accepted and relative < relative_tolerance:
            converged = True
            break
        if rejected >= 30:
            break
        previous = best[0]

    final_model = render_hapke_stack(p, q, w, sun, view, parameters, reflectance_model)
    return ExtendedSFSResult(
        dem=z, p=p, q=q, single_scattering_albedo=w,
        modeled_images=final_model, valid_mask=valid_mask,
        history=tuple(history), converged=converged,
    )
