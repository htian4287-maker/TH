"""Extended multi-image photoclinometry (PHCL).

This module implements the structure of Eq. (21)--(41) in Grumpe et al.
(2014): a reflectance residual over all images plus a Gaussian-low-pass
constraint toward the gradients of a coarse DEM.  The per-pixel 2x2
Gauss--Newton blocks retain the p/q cross term.  Single-scattering albedo is
estimated alternately at every pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import photometric_angles, slopes_to_normals
from .reflectance import (
    HapkeParameters,
    ReflectanceModel,
    hapke_reflectance,
)


@dataclass(frozen=True)
class PHCLResult:
    p: np.ndarray
    q: np.ndarray
    single_scattering_albedo: np.ndarray
    modeled_images: np.ndarray
    valid_mask: np.ndarray
    history: tuple[dict[str, float], ...]
    converged: bool


def reference_slopes(
    dem: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centered p=dz/dx and q=dz/dy from a DEM."""
    if pixel_size_x <= 0.0 or pixel_size_y <= 0.0:
        raise ValueError("Pixel sizes must be positive")
    q, p = np.gradient(
        np.asarray(dem, dtype=np.float64), pixel_size_y, pixel_size_x
    )
    return p, q


def centered_slopes_to_edges(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average centered slopes onto forward-difference edges."""
    p_values = np.asarray(p, dtype=np.float64)
    q_values = np.asarray(q, dtype=np.float64)
    if p_values.shape != q_values.shape or p_values.ndim != 2:
        raise ValueError("Centered p and q must be equal-sized 2-D arrays")
    return (
        0.5 * (p_values[:, :-1] + p_values[:, 1:]),
        0.5 * (q_values[:-1, :] + q_values[1:, :]),
    )


def _geometry_on_grid(
    directions: np.ndarray,
    image_count: int,
    shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    if values.shape == (image_count, 3):
        return np.broadcast_to(values[:, None, None, :], (image_count, *shape, 3))
    if values.shape == (image_count, *shape, 3):
        return values
    raise ValueError(
        f"Expected geometry shape {(image_count, 3)} or {(image_count, *shape, 3)}, "
        f"got {values.shape}"
    )


def render_hapke_stack(
    p: np.ndarray,
    q: np.ndarray,
    single_scattering_albedo: np.ndarray,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    parameters: HapkeParameters,
    reflectance_model: ReflectanceModel = "imsa",
) -> np.ndarray:
    """Render one selectable Hapke reflectance image per observation geometry."""
    p_values, q_values, w = np.broadcast_arrays(
        np.asarray(p, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
        np.asarray(single_scattering_albedo, dtype=np.float64),
    )
    if p_values.ndim != 2:
        raise ValueError("Slope and albedo fields must be two-dimensional")
    image_count = np.asarray(sun_directions).shape[0]
    sun = _geometry_on_grid(sun_directions, image_count, p_values.shape)
    view = _geometry_on_grid(view_directions, image_count, p_values.shape)
    normals = slopes_to_normals(p_values, q_values)
    normal_stack = np.broadcast_to(normals[None, ...], sun.shape)
    mu0, mu, phase = photometric_angles(normal_stack, sun, view)
    return hapke_reflectance(
        mu0,
        mu,
        phase,
        replace(parameters, single_scattering_albedo=w[None, ...]),
        multiple_scattering=reflectance_model,
    )


def render_imsa_stack(
    p: np.ndarray,
    q: np.ndarray,
    single_scattering_albedo: np.ndarray,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    parameters: HapkeParameters,
) -> np.ndarray:
    """Backward-compatible IMSA renderer used by the first three stages."""
    return render_hapke_stack(
        p,
        q,
        single_scattering_albedo,
        sun_directions,
        view_directions,
        parameters,
        "imsa",
    )


def _lowpass(values: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px == 0.0:
        return values
    return gaussian_filter(values, sigma_px, mode="reflect", truncate=4.0)


def _lowpass_normal_diagonal(sigma_px: float) -> float:
    """Diagonal of G.T G for a normalized separable Gaussian kernel."""
    if sigma_px == 0.0:
        return 1.0
    radius = int(4.0 * sigma_px + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma_px) ** 2)
    kernel /= kernel.sum()
    return float(np.sum(kernel * kernel) ** 2)


def _objective(
    model: np.ndarray,
    observed: np.ndarray,
    usable: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    weight: float,
    sigma_px: float,
) -> tuple[float, float, float]:
    residual = np.where(usable, model - observed, 0.0)
    data = 0.5 * float(np.sum(residual * residual))
    prior_p = _lowpass(p - p0, sigma_px)
    prior_q = _lowpass(q - q0, sigma_px)
    prior = 0.5 * weight * float(np.sum(prior_p * prior_p + prior_q * prior_q))
    return data + prior, data, prior


def _update_albedo(
    observed: np.ndarray,
    usable: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    sun: np.ndarray,
    view: np.ndarray,
    parameters: HapkeParameters,
    reflectance_model: ReflectanceModel,
    sigma_px: float = 0.0,
    iterations: int = 5,
) -> np.ndarray:
    estimate = np.clip(w.copy(), 0.02, 0.98)
    epsilon = 2.0e-3
    for _ in range(iterations):
        model = render_hapke_stack(
            p, q, estimate, sun, view, parameters, reflectance_model
        )
        plus = render_hapke_stack(
            p, q, np.clip(estimate + epsilon, 0.02, 0.98), sun, view,
            parameters, reflectance_model
        )
        minus = render_hapke_stack(
            p, q, np.clip(estimate - epsilon, 0.02, 0.98), sun, view,
            parameters, reflectance_model
        )
        derivative = (plus - minus) / (2.0 * epsilon)
        active = usable & np.isfinite(model) & np.isfinite(derivative)
        numerator = np.sum(
            np.where(active, derivative * (observed - model), 0.0), axis=0
        )
        denominator = np.sum(
            np.where(active, derivative * derivative, 0.0), axis=0
        )
        # Eq. (42) estimates the albedo at each pixel from a Gaussian-weighted
        # neighbourhood.  Filtering the accumulated normal-equation numerator
        # and denominator is the discrete equivalent of that weighted fit.
        if sigma_px > 0.0:
            numerator = _lowpass(numerator, sigma_px)
            denominator = _lowpass(denominator, sigma_px)
        estimate = np.clip(
            estimate + numerator / np.maximum(denominator, 1.0e-12), 0.02, 0.98
        )
    return estimate


def estimate_phcl(
    observed_images: np.ndarray,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    reference_dem: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    parameters: HapkeParameters,
    reflectance_model: ReflectanceModel = "imsa",
    reference_p: np.ndarray | None = None,
    reference_q: np.ndarray | None = None,
    warm_start_p: np.ndarray | None = None,
    warm_start_q: np.ndarray | None = None,
    warm_start_albedo: np.ndarray | None = None,
    initial_damping: float = 1.0e-4,
    previous_error: float = np.inf,
    consecutive_rejections: int = 0,
    iteration_offset: int = 0,
    dem_weight: float = 2.0e-4,
    dem_sigma_px: float = 15.0,
    albedo_sigma_px: float = 0.0,
    initial_albedo: float = 0.4,
    max_iterations: int = 80,
    relative_tolerance: float = 1.0e-6,
    max_rejections: int = 50,
    min_observations: int = 3,
) -> PHCLResult:
    """Estimate p, q and pixel-wise w from co-registered reflectance images."""
    observed = np.asarray(observed_images, dtype=np.float64)
    reference = np.asarray(reference_dem, dtype=np.float64)
    if observed.ndim != 3 or reference.shape != observed.shape[1:]:
        raise ValueError("Images must have shape (image,row,column) matching the DEM")
    if not np.isfinite(reference).all():
        raise ValueError("Reference DEM must be finite")
    if dem_weight < 0.0 or dem_sigma_px < 0.0 or albedo_sigma_px < 0.0:
        raise ValueError("DEM weight and sigma values must be non-negative")
    image_count = observed.shape[0]
    sun = _geometry_on_grid(sun_directions, image_count, reference.shape)
    view = _geometry_on_grid(view_directions, image_count, reference.shape)
    if (reference_p is None) != (reference_q is None):
        raise ValueError("reference_p and reference_q must be provided together")
    if reference_p is None:
        p0, q0 = reference_slopes(reference, pixel_size_x, pixel_size_y)
    else:
        p0 = np.asarray(reference_p, dtype=np.float64)
        q0 = np.asarray(reference_q, dtype=np.float64)
        if p0.shape != reference.shape or q0.shape != reference.shape:
            raise ValueError("Reference slopes must match the DEM grid")
        if not np.isfinite(p0).all() or not np.isfinite(q0).all():
            raise ValueError("Reference slopes must be finite")
    if (warm_start_p is None) != (warm_start_q is None):
        raise ValueError("warm_start_p and warm_start_q must be provided together")
    p = p0.copy() if warm_start_p is None else np.asarray(warm_start_p, dtype=np.float64).copy()
    q = q0.copy() if warm_start_q is None else np.asarray(warm_start_q, dtype=np.float64).copy()
    if p.shape != reference.shape or q.shape != reference.shape:
        raise ValueError("Warm-start slopes must match the DEM grid")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Warm-start slopes must be finite")
    if warm_start_albedo is None:
        w = np.full(reference.shape, initial_albedo, dtype=np.float64)
    else:
        w = np.asarray(warm_start_albedo, dtype=np.float64).copy()
        if w.shape != reference.shape or not np.isfinite(w).all():
            raise ValueError("Warm-start albedo must be finite and match the DEM grid")
        w = np.clip(w, 0.02, 0.98)
    if initial_damping <= 0.0 or consecutive_rejections < 0 or iteration_offset < 0:
        raise ValueError("Invalid warm-start optimizer state")
    # Eligibility must remain tied to the reference surface so continuation
    # uses exactly the same fixed observation mask as the original cold run.
    eligibility_w = np.full(reference.shape, initial_albedo, dtype=np.float64)
    base_model = render_hapke_stack(
        p0, q0, eligibility_w, sun, view, parameters, reflectance_model
    )
    usable = np.isfinite(observed) & np.isfinite(base_model) & (observed > 0.0)
    if min_observations < 1 or min_observations > image_count:
        raise ValueError("min_observations must lie between one and image_count")
    valid_mask = usable.sum(axis=0) >= min_observations
    usable &= valid_mask[None, ...]
    if valid_mask.sum() == 0:
        raise ValueError(
            f"No pixels have at least {min_observations} usable observations"
        )

    damping = float(initial_damping)
    prior_diagonal = dem_weight * _lowpass_normal_diagonal(dem_sigma_px)
    history: list[dict[str, float]] = []
    rejected = int(consecutive_rejections)
    converged = False
    previous_error = float(previous_error)
    slope_epsilon = 1.0e-4

    for iteration in range(max_iterations):
        w = _update_albedo(
            observed,
            usable,
            p,
            q,
            w,
            sun,
            view,
            parameters,
            reflectance_model,
            sigma_px=albedo_sigma_px,
        )
        model = render_hapke_stack(
            p, q, w, sun, view, parameters, reflectance_model
        )
        current_error, data_error, prior_error = _objective(
            model, observed, usable, p, q, p0, q0, dem_weight, dem_sigma_px
        )
        model_p_plus = render_hapke_stack(
            p + slope_epsilon, q, w, sun, view, parameters, reflectance_model
        )
        model_p_minus = render_hapke_stack(
            p - slope_epsilon, q, w, sun, view, parameters, reflectance_model
        )
        model_q_plus = render_hapke_stack(
            p, q + slope_epsilon, w, sun, view, parameters, reflectance_model
        )
        model_q_minus = render_hapke_stack(
            p, q - slope_epsilon, w, sun, view, parameters, reflectance_model
        )
        jp = (model_p_plus - model_p_minus) / (2.0 * slope_epsilon)
        jq = (model_q_plus - model_q_minus) / (2.0 * slope_epsilon)
        active = usable & np.isfinite(model) & np.isfinite(jp) & np.isfinite(jq)
        residual = np.where(active, model - observed, 0.0)
        jp_active = np.where(active, jp, 0.0)
        jq_active = np.where(active, jq, 0.0)

        gradient_p = np.sum(jp_active * residual, axis=0)
        gradient_q = np.sum(jq_active * residual, axis=0)
        gradient_p += dem_weight * _lowpass(
            _lowpass(p - p0, dem_sigma_px), dem_sigma_px
        )
        gradient_q += dem_weight * _lowpass(
            _lowpass(q - q0, dem_sigma_px), dem_sigma_px
        )
        hpp = np.sum(jp_active * jp_active, axis=0) + prior_diagonal + damping
        hqq = np.sum(jq_active * jq_active, axis=0) + prior_diagonal + damping
        hpq = np.sum(jp_active * jq_active, axis=0)
        determinant = np.maximum(hpp * hqq - hpq * hpq, 1.0e-15)
        delta_p = (-hqq * gradient_p + hpq * gradient_q) / determinant
        delta_q = (hpq * gradient_p - hpp * gradient_q) / determinant
        delta_p = np.clip(delta_p, -0.25, 0.25)
        delta_q = np.clip(delta_q, -0.25, 0.25)
        candidate_p = np.where(valid_mask, np.clip(p + delta_p, -3.0, 3.0), p0)
        candidate_q = np.where(valid_mask, np.clip(q + delta_q, -3.0, 3.0), q0)
        candidate_model = render_hapke_stack(
            candidate_p, candidate_q, w, sun, view, parameters,
            reflectance_model
        )
        candidate_error, candidate_data, candidate_prior = _objective(
            candidate_model,
            observed,
            usable,
            candidate_p,
            candidate_q,
            p0,
            q0,
            dem_weight,
            dem_sigma_px,
        )
        accepted = candidate_error < current_error
        if accepted:
            p, q, model = candidate_p, candidate_q, candidate_model
            new_error, data_error, prior_error = (
                candidate_error,
                candidate_data,
                candidate_prior,
            )
            damping = max(damping * 0.2, 1.0e-10)
            rejected = 0
        else:
            new_error = current_error
            damping = min(damping * 1.5, 1.0e8)
            rejected += 1
        relative_change = (
            abs(previous_error - new_error) / max(abs(previous_error), 1.0)
            if np.isfinite(previous_error)
            else np.inf
        )
        history.append(
            {
                "iteration": float(iteration_offset + iteration + 1),
                "total_error": float(new_error),
                "data_error": float(data_error),
                "dem_error": float(prior_error),
                "damping": float(damping),
                "accepted": float(accepted),
                "relative_change": float(relative_change),
            }
        )
        if accepted and relative_change < relative_tolerance:
            converged = True
            break
        if rejected >= max_rejections:
            break
        previous_error = new_error

    final_model = render_hapke_stack(
        p, q, w, sun, view, parameters, reflectance_model
    )
    return PHCLResult(
        p=p,
        q=q,
        single_scattering_albedo=w,
        modeled_images=final_model,
        valid_mask=valid_mask,
        history=tuple(history),
        converged=converged,
    )
