"""Photometric and raster operations for the Wu et al. (2016) SAfS reproduction.

Coordinates are local east, north, up (ENU). Raster columns increase eastward,
raster rows increase southward, and azimuth is clockwise from north.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


EPS = 1.0e-12


def normalize_directions(vectors: np.ndarray) -> np.ndarray:
    """Normalize one direction or a per-pixel direction field."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("Direction arrays must end in three ENU components")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("Direction vector has zero length")
    return values / norms


def direction_from_azimuth_zenith(
    azimuth_degrees: np.ndarray | float,
    zenith_degrees: np.ndarray | float,
) -> np.ndarray:
    """Create ENU directions; zenith is measured down from local vertical."""
    azimuth = np.deg2rad(azimuth_degrees)
    zenith = np.deg2rad(zenith_degrees)
    azimuth, zenith = np.broadcast_arrays(azimuth, zenith)
    return normalize_directions(
        np.stack(
            (
                np.sin(azimuth) * np.sin(zenith),
                np.cos(azimuth) * np.sin(zenith),
                np.cos(zenith),
            ),
            axis=-1,
        )
    )


def direction_from_azimuth_elevation(
    azimuth_degrees: np.ndarray | float,
    elevation_degrees: np.ndarray | float,
) -> np.ndarray:
    return direction_from_azimuth_zenith(
        azimuth_degrees, 90.0 - np.asarray(elevation_degrees)
    )


def cell_slopes(
    dem: np.ndarray, pixel_size_x: float, pixel_size_y: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply paper Eq. (2) to every 2x2 height cell.

    The returned values are p=dz/deast and q=dz/dnorth. The signs are adapted
    to north-up rasters whose row coordinate grows southward.
    """
    z00 = dem[:-1, :-1]
    z01 = dem[:-1, 1:]
    z10 = dem[1:, :-1]
    z11 = dem[1:, 1:]
    east = ((z01 + z11) - (z00 + z10)) / (2.0 * pixel_size_x)
    north = -((z10 + z11) - (z00 + z01)) / (2.0 * pixel_size_y)
    return east, north


def pixel_slopes(
    dem: np.ndarray, pixel_size_x: float, pixel_size_y: float
) -> tuple[np.ndarray, np.ndarray]:
    row_derivative, column_derivative = np.gradient(np.asarray(dem, dtype=np.float64))
    return column_derivative / pixel_size_x, -row_derivative / pixel_size_y


def normals_from_slopes(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    scale = np.sqrt(1.0 + east * east + north * north)
    return np.stack((-east / scale, -north / scale, 1.0 / scale), axis=-1)


def lunar_lambert_reflectance(
    normals: np.ndarray,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    lunar_lambert_l: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the configurable Lunar-Lambert mixture.

    G = 2 L mu0/(mu0+mu) + (1-L) mu0. Wu et al. name the model but do
    not publish the adopted phase-dependent coefficients, so L remains an
    explicit sensitivity parameter.
    """
    if not 0.0 <= lunar_lambert_l <= 1.0:
        raise ValueError("lunar_lambert_l must lie in [0, 1]")
    normals = normalize_directions(normals)
    sun = normalize_directions(sun_directions)
    view = normalize_directions(view_directions)
    mu0 = np.sum(normals * sun, axis=-1)
    mu = np.sum(normals * view, axis=-1)
    denominator = np.maximum(mu0 + mu, EPS)
    reflectance = (
        2.0 * lunar_lambert_l * mu0 / denominator
        + (1.0 - lunar_lambert_l) * mu0
    )
    valid = (mu0 > 0.0) & (mu > 0.0)
    return np.where(valid, np.maximum(reflectance, 0.0), 0.0), mu0, mu


def _directions_to_cells(directions: np.ndarray) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim == 1:
        return normalize_directions(values)
    cells = 0.25 * (
        values[:-1, :-1]
        + values[:-1, 1:]
        + values[1:, :-1]
        + values[1:, 1:]
    )
    return normalize_directions(cells)


def render_reflectance(
    dem: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    lunar_lambert_l: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    east, north = pixel_slopes(dem, pixel_size_x, pixel_size_y)
    reflectance, mu0, mu = lunar_lambert_reflectance(
        normals_from_slopes(east, north),
        sun_directions,
        view_directions,
        lunar_lambert_l,
    )
    return reflectance, (mu0 > 0.0) & (mu > 0.0)


def render_cell_reflectance(
    dem: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    lunar_lambert_l: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    east, north = cell_slopes(dem, pixel_size_x, pixel_size_y)
    reflectance, mu0, mu = lunar_lambert_reflectance(
        normals_from_slopes(east, north),
        _directions_to_cells(sun_directions),
        _directions_to_cells(view_directions),
        lunar_lambert_l,
    )
    return reflectance, east, north, (mu0 > 0.0) & (mu > 0.0)


def four_corner_mean(values: np.ndarray) -> np.ndarray:
    return 0.25 * (
        values[:-1, :-1]
        + values[:-1, 1:]
        + values[1:, :-1]
        + values[1:, 1:]
    )


def four_corner_all(mask: np.ndarray) -> np.ndarray:
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]


def node_sum_from_cells(cell_values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float64)
    result[:-1, :-1] += cell_values
    result[:-1, 1:] += cell_values
    result[1:, :-1] += cell_values
    result[1:, 1:] += cell_values
    return result


def node_mean_from_cells(cell_values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    total = node_sum_from_cells(cell_values, shape)
    counts = node_sum_from_cells(np.ones(cell_values.shape, dtype=np.float64), shape)
    return total / np.maximum(counts, 1.0)


def _resize_finite(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(array, dtype=np.float64)
    if source.shape == shape:
        return source.copy()
    if source.ndim != 2:
        raise ValueError("Only two-dimensional arrays can be resized")
    source_y = np.linspace(0.0, 1.0, source.shape[0])
    source_x = np.linspace(0.0, 1.0, source.shape[1])
    target_y = np.linspace(0.0, 1.0, shape[0])
    target_x = np.linspace(0.0, 1.0, shape[1])
    horizontal = np.empty((source.shape[0], shape[1]), dtype=np.float64)
    for row in range(source.shape[0]):
        horizontal[row] = np.interp(target_x, source_x, source[row])
    output = np.empty(shape, dtype=np.float64)
    for column in range(shape[1]):
        output[:, column] = np.interp(target_y, source_y, horizontal[:, column])
    return output


def resize_bilinear(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(source)
    if finite.all():
        return _resize_finite(source, shape)
    values = _resize_finite(np.where(finite, source, 0.0), shape)
    weights = _resize_finite(finite.astype(np.float64), shape)
    return np.where(weights > 1.0e-6, values / np.maximum(weights, EPS), np.nan)


def resize_directions(directions: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim == 1:
        return normalize_directions(values)
    if values.shape[:2] == shape:
        return normalize_directions(values.copy())
    resized = np.stack(
        [resize_bilinear(values[..., index], shape) for index in range(3)], axis=-1
    )
    return normalize_directions(resized)


def local_mean(
    array: np.ndarray, window: int, valid: np.ndarray | None = None
) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be positive")
    source = np.asarray(array, dtype=np.float64)
    mask = np.isfinite(source)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    values = uniform_filter(np.where(mask, source, 0.0), size=window, mode="nearest")
    weights = uniform_filter(mask.astype(np.float64), size=window, mode="nearest")
    return np.where(weights > 1.0e-6, values / np.maximum(weights, EPS), np.nan)


def estimate_piecewise_albedo(
    image: np.ndarray,
    reflectance: np.ndarray,
    valid: np.ndarray,
    window: int,
    minimum: float,
    maximum: float,
    reflectance_smoothing_window: int = 3,
) -> tuple[np.ndarray, float]:
    """Operationalize paper Eqs. (4)-(8) in the log domain."""
    smooth_g = local_mean(reflectance, reflectance_smoothing_window, valid)
    mask = (
        valid
        & np.isfinite(image)
        & np.isfinite(smooth_g)
        & (image > EPS)
        & (smooth_g > EPS)
    )
    log_albedo = np.full(image.shape, np.nan, dtype=np.float64)
    log_albedo[mask] = np.log(image[mask]) - np.log(smooth_g[mask])
    local_log_albedo = local_mean(log_albedo, window, mask)
    if not np.isfinite(local_log_albedo[mask]).any():
        return np.ones_like(image), 1.0
    exposure_log = float(np.nanmedian(local_log_albedo[mask]))
    albedo = np.exp(local_log_albedo - exposure_log)
    albedo = np.clip(albedo, minimum, maximum)
    albedo[~np.isfinite(albedo)] = 1.0
    return albedo, float(np.exp(exposure_log))
