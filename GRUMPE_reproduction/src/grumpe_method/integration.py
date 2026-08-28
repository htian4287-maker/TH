"""Gradient-to-elevation recovery with low-pass absolute-depth control.

The objective is Eq. (4) of Grumpe & Woehler (2014):

  1/2 ||D z - (p,q)||^2 + s/2 ||G_sigma * (z-z_dem)||^2.

The paper derives a pixel-wise relaxation update.  Here the same quadratic
functional is solved directly with matrix-free conjugate gradients.  This
changes the numerical linear solver, not the optimum being estimated, and
provides a stable reference against which a literal relaxation port can be
checked later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg


@dataclass(frozen=True)
class IntegrationResult:
    dem: np.ndarray
    converged: bool
    iterations: int
    relative_residual: float
    gradient_energy: float
    depth_energy: float


def forward_gradient(
    dem: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward differences p=dz/dx and q=dz/dy on cell edges."""
    z = np.asarray(dem, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("DEM must be a two-dimensional array of at least 2x2")
    if pixel_size_x <= 0.0 or pixel_size_y <= 0.0:
        raise ValueError("Pixel sizes must be positive")
    return np.diff(z, axis=1) / pixel_size_x, np.diff(z, axis=0) / pixel_size_y


def gradient_adjoint(
    p: np.ndarray,
    q: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
) -> np.ndarray:
    """Apply the transpose of the forward-difference operator."""
    px = np.asarray(p, dtype=np.float64)
    qy = np.asarray(q, dtype=np.float64)
    rows, columns_minus_one = px.shape
    columns = columns_minus_one + 1
    if qy.shape != (rows - 1, columns):
        raise ValueError("Expected p=(rows, cols-1), q=(rows-1, cols)")
    result = np.zeros((rows, columns), dtype=np.float64)
    result[:, :-1] -= px / pixel_size_x
    result[:, 1:] += px / pixel_size_x
    result[:-1, :] -= qy / pixel_size_y
    result[1:, :] += qy / pixel_size_y
    return result


def _lowpass(values: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px == 0.0:
        return values
    return gaussian_filter(values, sigma=sigma_px, mode="reflect", truncate=4.0)


def integrate_gradients(
    p: np.ndarray,
    q: np.ndarray,
    reference_dem: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    depth_weight: float,
    lowpass_sigma_px: float,
    rtol: float = 1.0e-9,
    max_iterations: int = 5000,
) -> IntegrationResult:
    """Minimize the Grumpe low-pass absolute-depth functional.

    Set ``depth_weight=0`` for Horn/Poisson integration.  Set
    ``lowpass_sigma_px=0`` with a positive weight for the Shao-style direct
    height constraint.  A tiny datum anchor is used only in the Poisson case.
    """
    if depth_weight < 0.0 or lowpass_sigma_px < 0.0:
        raise ValueError("Weights and Gaussian sigma must be non-negative")
    reference = np.asarray(reference_dem, dtype=np.float64)
    if reference.ndim != 2 or not np.isfinite(reference).all():
        raise ValueError("Reference DEM must be a finite two-dimensional array")
    expected_p, expected_q = forward_gradient(reference, pixel_size_x, pixel_size_y)
    p_values = np.asarray(p, dtype=np.float64)
    q_values = np.asarray(q, dtype=np.float64)
    if p_values.shape != expected_p.shape or q_values.shape != expected_q.shape:
        raise ValueError("Slope fields do not match the reference DEM grid")
    if not np.isfinite(p_values).all() or not np.isfinite(q_values).all():
        raise ValueError("Slope fields must be finite")

    anchor = 1.0e-10 if depth_weight == 0.0 else 0.0
    shape = reference.shape
    size = reference.size

    def apply_normal(vector: np.ndarray) -> np.ndarray:
        surface = vector.reshape(shape)
        sx, sy = forward_gradient(surface, pixel_size_x, pixel_size_y)
        result = gradient_adjoint(sx, sy, pixel_size_x, pixel_size_y)
        if depth_weight > 0.0:
            result += depth_weight * _lowpass(
                _lowpass(surface, lowpass_sigma_px), lowpass_sigma_px
            )
        if anchor > 0.0:
            result += anchor * surface
        return result.ravel()

    rhs = gradient_adjoint(p_values, q_values, pixel_size_x, pixel_size_y)
    if depth_weight > 0.0:
        rhs += depth_weight * _lowpass(
            _lowpass(reference, lowpass_sigma_px), lowpass_sigma_px
        )
    if anchor > 0.0:
        rhs += anchor * reference

    iterations = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    operator = LinearOperator((size, size), matvec=apply_normal, dtype=np.float64)
    solution, info = cg(
        operator,
        rhs.ravel(),
        x0=reference.ravel(),
        rtol=rtol,
        atol=0.0,
        maxiter=max_iterations,
        callback=callback,
    )
    dem = solution.reshape(shape)
    residual = np.linalg.norm(apply_normal(solution) - rhs.ravel()) / max(
        np.linalg.norm(rhs.ravel()), 1.0
    )
    dem_p, dem_q = forward_gradient(dem, pixel_size_x, pixel_size_y)
    gradient_energy = 0.5 * (
        np.mean((dem_p - p_values) ** 2) + np.mean((dem_q - q_values) ** 2)
    )
    depth_energy = 0.5 * depth_weight * np.mean(
        _lowpass(dem - reference, lowpass_sigma_px) ** 2
    )
    return IntegrationResult(
        dem=dem,
        converged=info == 0,
        iterations=iterations,
        relative_residual=float(residual),
        gradient_energy=float(gradient_energy),
        depth_energy=float(depth_energy),
    )

