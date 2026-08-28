"""Local Cartesian geometry used by the photometric reconstruction.

The package uses a right-handed frame: x=east, y=north, z=up.  Surface
slopes are p=dz/dx and q=dz/dy, so a unit normal is proportional to
[-p, -q, 1].
"""

from __future__ import annotations

import numpy as np


def normalize(vectors: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Return unit vectors and reject zero-length inputs."""
    values = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    if np.any(norm <= 1.0e-15):
        raise ValueError("Cannot normalize a zero-length vector")
    return values / norm


def slopes_to_normals(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Convert p=dz/dx and q=dz/dy into normals with shape (..., 3)."""
    p_values, q_values = np.broadcast_arrays(
        np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    )
    normals = np.stack((-p_values, -q_values, np.ones_like(p_values)), axis=-1)
    return normalize(normals)


def normals_to_slopes(normals: np.ndarray, *, min_up: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    """Convert upward-facing normals into p=dz/dx and q=dz/dy."""
    n = normalize(np.asarray(normals, dtype=np.float64))
    up = n[..., 2]
    if np.any(up <= min_up):
        raise ValueError("Normals must face upward")
    return -n[..., 0] / up, -n[..., 1] / up


def direction_from_azimuth_zenith(
    azimuth_degrees: np.ndarray | float,
    zenith_degrees: np.ndarray | float,
) -> np.ndarray:
    """Return ENU directions; azimuth is clockwise from north."""
    azimuth = np.deg2rad(azimuth_degrees)
    zenith = np.deg2rad(zenith_degrees)
    azimuth, zenith = np.broadcast_arrays(azimuth, zenith)
    return np.stack(
        (
            np.sin(azimuth) * np.sin(zenith),
            np.cos(azimuth) * np.sin(zenith),
            np.cos(zenith),
        ),
        axis=-1,
    )


def photometric_angles(
    normals: np.ndarray,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mu0=cos(incidence), mu=cos(emission), and phase angle.

    Geometry arrays broadcast against the leading dimensions of ``normals``.
    The phase angle is between the surface-to-Sun and surface-to-sensor
    vectors and therefore does not depend on the surface normal.
    """
    n = normalize(normals)
    s = normalize(sun_directions)
    v = normalize(view_directions)
    mu0 = np.sum(n * s, axis=-1)
    mu = np.sum(n * v, axis=-1)
    phase = np.arccos(np.clip(np.sum(s * v, axis=-1), -1.0, 1.0))
    return mu0, mu, phase

