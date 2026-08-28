"""Paper-style iterative relaxation for low-pass-constrained integration.

This is a direct implementation of the common special case of Eq. (31)--(34)
in Grumpe & Woehler (2014): a rotationally symmetric Gaussian filter and
constant (possibly unequal) x/y pixel extents.  Under these assumptions c2
vanishes and every non-singular pixel has one update z_c = -c0/c1.

The paper's terminology distinguishes an update *step* from a successful
*iteration*: a step is an iteration only when it lowers the best objective.
The stopping counters below preserve that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, gaussian_filter


@dataclass(frozen=True)
class RelaxationResult:
    dem: np.ndarray
    current_dem: np.ndarray
    successful_iterations: int
    total_updates: int
    steps_since_improvement: int
    best_error: float
    stop_reason: str
    history: tuple[dict[str, float], ...]
    singular_pixels_last_update: int


def gaussian_kernel(sigma_px: float, *, truncate: float = 4.0) -> np.ndarray:
    """Return the normalized, rotationally symmetric filter matrix F."""
    if sigma_px <= 0.0:
        raise ValueError("The paper relaxation solver requires sigma_px > 0")
    radius = max(1, int(truncate * sigma_px + 0.5))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    one_dimensional = np.exp(-0.5 * (coordinates / sigma_px) ** 2)
    one_dimensional /= one_dimensional.sum()
    kernel = np.outer(one_dimensional, one_dimensional)
    kernel /= kernel.sum()
    return kernel


def _sigma_from_kernel(kernel: np.ndarray) -> float:
    """Recover the generating Gaussian sigma from center/adjacent ratio."""
    center_y, center_x = np.array(kernel.shape) // 2
    center = float(kernel[center_y, center_x])
    adjacent = float(kernel[center_y, center_x + 1])
    ratio = adjacent / center
    if not 0.0 < ratio < 1.0:
        raise ValueError("Kernel is not a non-degenerate sampled Gaussian")
    return float(np.sqrt(-1.0 / (2.0 * np.log(ratio))))


def _centered_gradient(
    values: np.ndarray, pixel_size_x: float, pixel_size_y: float
) -> tuple[np.ndarray, np.ndarray]:
    edge_order = 2 if min(values.shape) >= 3 else 1
    q, p = np.gradient(values, pixel_size_y, pixel_size_x, edge_order=edge_order)
    return p, q


def _divergence(
    p: np.ndarray, q: np.ndarray, pixel_size_x: float, pixel_size_y: float
) -> np.ndarray:
    edge_order = 2 if min(p.shape) >= 3 else 1
    px = np.gradient(p, pixel_size_x, axis=1, edge_order=edge_order)
    qy = np.gradient(q, pixel_size_y, axis=0, edge_order=edge_order)
    return px + qy


def _neighbours(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return l, r, u, d, 2l, 2r, 2u, 2d using reflected ghost cells."""
    padded = np.pad(values, 2, mode="reflect")
    return (
        padded[2:-2, 1:-3],
        padded[2:-2, 3:-1],
        padded[3:-1, 2:-2],
        padded[1:-3, 2:-2],
        padded[2:-2, 0:-4],
        padded[2:-2, 4:],
        padded[4:, 2:-2],
        padded[0:-4, 2:-2],
    )


def _apply_neumann_boundary(
    surface: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
) -> None:
    """Apply Eq. (9) on the rectangular outer boundary in-place."""
    surface[1:-1, 0] = surface[1:-1, 1] - p[1:-1, 0] * pixel_size_x
    surface[1:-1, -1] = surface[1:-1, -2] + p[1:-1, -1] * pixel_size_x
    surface[0, 1:-1] = surface[1, 1:-1] - q[0, 1:-1] * pixel_size_y
    surface[-1, 1:-1] = surface[-2, 1:-1] + q[-1, 1:-1] * pixel_size_y
    surface[0, 0] = 0.5 * (
        surface[0, 1] - p[0, 0] * pixel_size_x
        + surface[1, 0] - q[0, 0] * pixel_size_y
    )
    surface[0, -1] = 0.5 * (
        surface[0, -2] + p[0, -1] * pixel_size_x
        + surface[1, -1] - q[0, -1] * pixel_size_y
    )
    surface[-1, 0] = 0.5 * (
        surface[-1, 1] - p[-1, 0] * pixel_size_x
        + surface[-2, 0] + q[-1, 0] * pixel_size_y
    )
    surface[-1, -1] = 0.5 * (
        surface[-1, -2] + p[-1, -1] * pixel_size_x
        + surface[-2, -1] + q[-1, -1] * pixel_size_y
    )


def relaxation_energy(
    surface: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    reference_dem: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    depth_weight: float,
    kernel: np.ndarray,
) -> tuple[float, float, float]:
    """Evaluate Eq. (4) with centered derivatives and the discrete F."""
    surface_p, surface_q = _centered_gradient(surface, pixel_size_x, pixel_size_y)
    integrability = 0.5 * float(
        np.sum((surface_p - p) ** 2 + (surface_q - q) ** 2)
    )
    sigma_px = _sigma_from_kernel(kernel)
    lowpass_difference = gaussian_filter(
        surface - reference_dem, sigma_px, mode="reflect", truncate=4.0
    )
    depth = 0.5 * depth_weight * float(np.sum(lowpass_difference**2))
    return integrability + depth, integrability, depth


def _interpolate_singular(
    candidate: np.ndarray,
    regular: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Implement the paper's expanding-neighbourhood singular-pixel fill."""
    result = candidate.copy()
    known = regular.copy()
    unresolved = ~known
    max_radius = max(result.shape)
    for radius in range(1, max_radius + 1):
        if not np.any(unresolved):
            break
        footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        footprint[radius, radius] = 0.0
        count = convolve(known.astype(np.float64), footprint, mode="constant", cval=0.0)
        total = convolve(np.where(known, result, 0.0), footprint, mode="constant", cval=0.0)
        fill = unresolved & (count > 0.0)
        result[fill] = total[fill] / count[fill]
        known[fill] = True
        unresolved &= ~fill
    # A completely flat gradient field has no regular pixel.  The paper notes
    # Poisson updating as an alternative; use it only for that residual case.
    result[unresolved] = fallback[unresolved]
    return result


def paper_relaxation_update(
    surface: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    reference_dem: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    depth_weight: float,
    kernel: np.ndarray,
    singular_tolerance: float = 1.0e-14,
) -> tuple[np.ndarray, int]:
    """Apply one simultaneous Eq. (31)--(34) update to the whole image."""
    z = np.asarray(surface, dtype=np.float64)
    reference = np.asarray(reference_dem, dtype=np.float64)
    rows, columns = z.shape
    if p.shape != z.shape or q.shape != z.shape or reference.shape != z.shape:
        raise ValueError("surface, p, q, and reference DEM must share one grid")
    if min(rows, columns) < 5:
        raise ValueError("Relaxation needs at least a 5x5 grid")
    center_y, center_x = np.array(kernel.shape) // 2
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Filter kernel dimensions must be odd")
    if not np.allclose(kernel, kernel[::-1, ::-1], atol=1.0e-14):
        raise ValueError("This implemented paper branch requires a symmetric filter")

    f0 = float(kernel[center_y, center_x])
    fl = float(kernel[center_y, center_x - 1])
    fr = float(kernel[center_y, center_x + 1])
    fu = float(kernel[center_y + 1, center_x])
    fd = float(kernel[center_y - 1, center_x])
    sigma_px = _sigma_from_kernel(kernel)

    zx, zy = _centered_gradient(z, pixel_size_x, pixel_size_y)
    denominator = zx * zx + zy * zy
    regular = denominator > singular_tolerance
    safe_denominator = np.where(regular, denominator, 1.0)
    zl, zr, zu, zd, z2l, z2r, z2u, z2d = _neighbours(z)
    zh = zl + zr
    zv = zu + zd
    # A Gaussian is separable.  Evaluate the full filter in O(N*sigma), then
    # remove the explicitly excluded coefficients from Eq. (18), (27), (28).
    # This is algebraically identical to convolving with each modified dense
    # 2-D kernel and is crucial for the paper's sigma=15 px grid search.
    full_z = gaussian_filter(z, sigma_px, mode="reflect", truncate=4.0)
    full_reference = gaussian_filter(reference, sigma_px, mode="reflect", truncate=4.0)
    full_zx = gaussian_filter(zx, sigma_px, mode="reflect", truncate=4.0)
    full_zy = gaussian_filter(zy, sigma_px, mode="reflect", truncate=4.0)
    f_minus_zero = full_z - f0 * z
    f_reference = full_reference
    zx_l, zx_r, _, _, _, _, _, _ = _neighbours(zx)
    _, _, zy_u, zy_d, _, _, _, _ = _neighbours(zy)
    f_minus_h_zx = full_zx - fl * zx_l - fr * zx_r
    f_minus_v_zy = full_zy - fu * zy_u - fd * zy_d

    # Eq. (30), second (z_c-independent) term.  Constant pixel extents imply
    # h_l=h_r=h_cx and h_u=h_d=h_cy.
    b_numerator = zx * (
        fl * z2r / (2.0 * pixel_size_x)
        - fr * z2l / (2.0 * pixel_size_x)
        + f_minus_h_zx
    )
    b_numerator += zy * (
        fd * z2u / (2.0 * pixel_size_y)
        - fu * z2d / (2.0 * pixel_size_y)
        + f_minus_v_zy
    )
    b = b_numerator / safe_denominator
    laplacian_coefficient = 2.0 * (
        pixel_size_x**2 + pixel_size_y**2
    ) / (pixel_size_x**2 * pixel_size_y**2)
    c1 = -depth_weight * f0 * b - laplacian_coefficient
    divergence = _divergence(p, q, pixel_size_x, pixel_size_y)
    c0 = -depth_weight * (f_minus_zero - f_reference) * b
    c0 += (
        pixel_size_y**2 * zh + pixel_size_x**2 * zv
    ) / (pixel_size_x**2 * pixel_size_y**2)
    c0 -= divergence
    candidate = np.where(regular, -c0 / np.where(np.abs(c1) > 1.0e-15, c1, -1.0), z)

    alpha_squared = (pixel_size_y / pixel_size_x) ** 2
    poisson = (
        alpha_squared * zh + zv - divergence * pixel_size_y**2
    ) / (2.0 + 2.0 * alpha_squared)
    candidate = _interpolate_singular(candidate, regular, poisson)
    _apply_neumann_boundary(candidate, p, q, pixel_size_x, pixel_size_y)
    return candidate, int((~regular).sum())


def solve_paper_relaxation(
    p: np.ndarray,
    q: np.ndarray,
    reference_dem: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    depth_weight: float,
    lowpass_sigma_px: float,
    max_successful_iterations: int = 100_000,
    max_steps_without_improvement: int = 500,
    max_total_updates: int | None = None,
    divergence_error: float = np.finfo(np.float64).max,
    relative_tolerance: float = 1.0e-10,
) -> RelaxationResult:
    """Run the paper flowchart and return the best accepted solution."""
    if depth_weight <= 0.0:
        raise ValueError("Use the Poisson solver for zero depth weight")
    reference = np.asarray(reference_dem, dtype=np.float64)
    p_values = np.asarray(p, dtype=np.float64)
    q_values = np.asarray(q, dtype=np.float64)
    if reference.ndim != 2 or p_values.shape != reference.shape or q_values.shape != reference.shape:
        raise ValueError("p, q, and reference DEM must be equal-sized 2-D arrays")
    if not np.isfinite(reference).all() or not np.isfinite(p_values).all() or not np.isfinite(q_values).all():
        raise ValueError("All inputs must be finite")
    kernel = gaussian_kernel(lowpass_sigma_px)
    current = reference.copy()
    best = current.copy()
    best_error, best_integrability, best_depth = relaxation_energy(
        current, p_values, q_values, reference,
        pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y,
        depth_weight=depth_weight, kernel=kernel,
    )
    history: list[dict[str, float]] = [
        {
            "update": 0.0,
            "successful_iteration": 0.0,
            "error": best_error,
            "integrability_error": best_integrability,
            "depth_error": best_depth,
            "improved": 1.0,
        }
    ]
    successful = 0
    stale_steps = 0
    total_limit = max_total_updates or (
        max_successful_iterations * max_steps_without_improvement
    )
    stop_reason = "max_total_updates"
    singular_count = 0

    for update_index in range(1, total_limit + 1):
        candidate, singular_count = paper_relaxation_update(
            current, p_values, q_values, reference,
            pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y,
            depth_weight=depth_weight, kernel=kernel,
        )
        error, integrability, depth = relaxation_energy(
            candidate, p_values, q_values, reference,
            pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y,
            depth_weight=depth_weight, kernel=kernel,
        )
        if not np.isfinite(error) or error >= divergence_error:
            stop_reason = "divergence_error"
            break
        improved = error < best_error
        relative_decrease = (
            (best_error - error) / max(abs(best_error), 1.0) if improved else 0.0
        )
        if improved:
            best = candidate.copy()
            best_error = error
            successful += 1
            stale_steps = 0
        else:
            stale_steps += 1
        current = candidate
        history.append(
            {
                "update": float(update_index),
                "successful_iteration": float(successful),
                "error": float(error),
                "integrability_error": float(integrability),
                "depth_error": float(depth),
                "improved": float(improved),
                "relative_decrease": float(relative_decrease),
            }
        )
        if improved and relative_decrease < relative_tolerance:
            stop_reason = "relative_tolerance"
            break
        if successful >= max_successful_iterations:
            stop_reason = "max_successful_iterations"
            break
        if stale_steps >= max_steps_without_improvement:
            stop_reason = "max_steps_without_improvement"
            break
    else:
        update_index = total_limit
    return RelaxationResult(
        dem=best,
        current_dem=current,
        successful_iterations=successful,
        total_updates=int(update_index),
        steps_since_improvement=stale_steps,
        best_error=float(best_error),
        stop_reason=stop_reason,
        history=tuple(history),
        singular_pixels_last_update=singular_count,
    )
