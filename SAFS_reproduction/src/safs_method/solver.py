"""Hierarchical shape-and-albedo-from-shading solver after Wu et al. (2016).

The paper does not publish source code or all numerical constants. This module
implements its stated hierarchy, Eq. (2) cell normals, Eqs. (4)-(8) local
albedo separation, Eq. (9) photometric/low-pass-normal objective, and
illumination-ordered relaxation. Every unpublished choice is configurable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .model import (
    EPS,
    cell_slopes,
    estimate_piecewise_albedo,
    four_corner_all,
    four_corner_mean,
    node_sum_from_cells,
    normalize_directions,
    render_cell_reflectance,
    render_reflectance,
    resize_bilinear,
    resize_directions,
)


@dataclass
class SafsConfig:
    iterations_per_level: int = 5
    maximum_pyramid_levels: int = 9
    albedo_window_final_px: int = 31
    reflectance_smoothing_final_px: int = 5
    lunar_lambert_l: float = 0.55
    reflectance_weight: float = 1.0
    normal_weight: float = 0.005
    finite_difference_step_m: float = 0.05
    maximum_height_step_m: float = 0.25
    newton_damping: float = 1.0e-6
    line_search_steps: int = 6
    convergence_tolerance_m: float = 1.0e-3
    shadow_threshold_normalized: float = 0.08
    shadow_percentile: float = 1.0
    albedo_minimum: float = 0.35
    albedo_maximum: float = 2.5
    preserve_vertical_datum: bool = True
    sweep_mode: str = "four_color"
    sequential_max_pixels: int = 65536

    def validate(self) -> None:
        if self.iterations_per_level < 1 or self.maximum_pyramid_levels < 1:
            raise ValueError("Iteration and pyramid-level counts must be positive")
        if self.albedo_window_final_px < 1 or self.reflectance_smoothing_final_px < 1:
            raise ValueError("Smoothing windows must be positive")
        if self.reflectance_weight <= 0.0 or self.normal_weight < 0.0:
            raise ValueError("Invalid objective weights")
        if self.finite_difference_step_m <= 0.0 or self.maximum_height_step_m <= 0.0:
            raise ValueError("Finite-difference and height steps must be positive")
        if self.newton_damping <= 0.0 or self.line_search_steps < 0:
            raise ValueError("Invalid Newton/line-search settings")
        if not 0.0 <= self.lunar_lambert_l <= 1.0:
            raise ValueError("lunar_lambert_l must lie in [0, 1]")
        if not 0.0 < self.albedo_minimum < self.albedo_maximum:
            raise ValueError("Invalid albedo bounds")
        if not 0.0 <= self.shadow_percentile < 50.0:
            raise ValueError("shadow_percentile must lie in [0, 50)")
        if self.sweep_mode not in {"four_color", "sequential"}:
            raise ValueError("sweep_mode must be four_color or sequential")


@dataclass
class SafsResult:
    dem: np.ndarray
    albedo: np.ndarray
    exposure: float
    modeled_intensity: np.ndarray
    valid_mask: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    pyramid_shapes: list[tuple[int, int]] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)


def pyramid_shapes(
    initial_shape: tuple[int, int],
    final_shape: tuple[int, int],
    maximum_levels: int,
) -> list[tuple[int, int]]:
    """Start on the actual coarse DEM grid and double toward the image grid."""
    if min(initial_shape) < 2 or min(final_shape) < 2:
        raise ValueError("Input grids are too small")
    if initial_shape[0] > final_shape[0] or initial_shape[1] > final_shape[1]:
        raise ValueError("The initial DEM cannot be finer than the image grid")
    shapes = [tuple(int(value) for value in initial_shape)]
    while shapes[-1] != final_shape:
        previous = shapes[-1]
        following = (
            min(final_shape[0], previous[0] * 2),
            min(final_shape[1], previous[1] * 2),
        )
        if following == previous:
            break
        shapes.append(following)
        if len(shapes) > maximum_levels:
            # Preserve the actual first and final grids; compress omitted middle levels.
            shapes = shapes[: maximum_levels - 1] + [final_shape]
            break
    return shapes


def _scaled_window(final_window: int, shape: tuple[int, int], final_shape: tuple[int, int]) -> int:
    scale = 0.5 * (shape[0] / final_shape[0] + shape[1] / final_shape[1])
    return max(1, int(round(final_window * scale)))


def _node_costs(
    dem: np.ndarray,
    reference_dem: np.ndarray,
    target_reflectance: np.ndarray,
    photometric_valid: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    config: SafsConfig,
) -> np.ndarray:
    predicted, east, north, illuminated = render_cell_reflectance(
        dem,
        pixel_size_x,
        pixel_size_y,
        sun_directions,
        view_directions,
        config.lunar_lambert_l,
    )
    active = photometric_valid & illuminated & np.isfinite(target_reflectance)
    photo_cells = np.where(
        active,
        config.reflectance_weight * (predicted - target_reflectance) ** 2,
        0.0,
    )
    photo_nodes = node_sum_from_cells(photo_cells, dem.shape)

    reference_east, reference_north = cell_slopes(
        reference_dem, pixel_size_x, pixel_size_y
    )
    # Eq. (3) and Sec. 2.3: after a factor-two upsampling, each parent DEM
    # cell owns a fixed 2x2 group of child cells.  The low-pass normal is the
    # arithmetic mean *inside that aligned group*.  It must not be the four
    # cells surrounding every height node: the latter cancels the derivative
    # of the centre height identically and silently disables the constraint.
    def aligned_group_mean(values: np.ndarray) -> np.ndarray:
        rows, columns = values.shape
        padded_rows = rows + rows % 2
        padded_columns = columns + columns % 2
        padded = np.pad(
            values,
            ((0, padded_rows - rows), (0, padded_columns - columns)),
            mode="edge",
        )
        groups = padded.reshape(padded_rows // 2, 2, padded_columns // 2, 2).mean(
            axis=(1, 3)
        )
        return np.repeat(np.repeat(groups, 2, axis=0), 2, axis=1)[:rows, :columns]

    low_east = aligned_group_mean(east)
    low_north = aligned_group_mean(north)
    reference_low_east = aligned_group_mean(reference_east)
    reference_low_north = aligned_group_mean(reference_north)
    normal_cells = config.normal_weight * (
        (low_east - reference_low_east) ** 2
        + (low_north - reference_low_north) ** 2
    )
    normal_nodes = node_sum_from_cells(normal_cells, dem.shape)
    return photo_nodes + normal_nodes


def _color_order(
    sun_directions: np.ndarray, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    sun = normalize_directions(sun_directions)
    if sun.ndim > 1:
        sun = normalize_directions(np.nanmedian(sun.reshape(-1, 3), axis=0))
    # Start in the same illumination quadrant, as stated in Sec. 2.3.
    row_parity = (shape[0] - 2) % 2 if sun[1] < 0.0 else 1
    column_parity = (shape[1] - 2) % 2 if sun[0] > 0.0 else 1
    return [
        (row_parity, column_parity),
        (row_parity, 1 - column_parity),
        (1 - row_parity, column_parity),
        (1 - row_parity, 1 - column_parity),
    ]


def _four_color_sweep(
    dem: np.ndarray,
    reference_dem: np.ndarray,
    target_reflectance: np.ndarray,
    photometric_valid: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    config: SafsConfig,
    optimization_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Vectorized Gauss-Seidel-compatible four-color relaxation."""
    updated = dem.copy()
    rows, columns = np.indices(updated.shape)
    interior = optimization_mask.copy()
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    epsilon = config.finite_difference_step_m
    maximum_step = 0.0
    accepted = 0
    attempted = 0

    for row_parity, column_parity in _color_order(sun_directions, updated.shape):
        selected = interior & (rows % 2 == row_parity) & (columns % 2 == column_parity)
        if not selected.any():
            continue
        base = _node_costs(
            updated, reference_dem, target_reflectance, photometric_valid,
            pixel_size_x, pixel_size_y, sun_directions, view_directions, config,
        )
        plus = updated.copy()
        minus = updated.copy()
        plus[selected] += epsilon
        minus[selected] -= epsilon
        plus_cost = _node_costs(
            plus, reference_dem, target_reflectance, photometric_valid,
            pixel_size_x, pixel_size_y, sun_directions, view_directions, config,
        )
        minus_cost = _node_costs(
            minus, reference_dem, target_reflectance, photometric_valid,
            pixel_size_x, pixel_size_y, sun_directions, view_directions, config,
        )
        gradient = (plus_cost - minus_cost) / (2.0 * epsilon)
        curvature = (plus_cost - 2.0 * base + minus_cost) / (epsilon * epsilon)
        denominator = np.where(
            curvature > config.newton_damping, curvature, config.newton_damping
        )
        step = np.clip(
            -gradient / denominator,
            -config.maximum_height_step_m,
            config.maximum_height_step_m,
        )
        step[~np.isfinite(step) | ~selected] = 0.0
        attempted += int(selected.sum())

        candidate = updated + step
        for _ in range(config.line_search_steps + 1):
            candidate_cost = _node_costs(
                candidate, reference_dem, target_reflectance, photometric_valid,
                pixel_size_x, pixel_size_y, sun_directions, view_directions, config,
            )
            improved = selected & np.isfinite(candidate_cost) & (candidate_cost <= base)
            unresolved = selected & ~improved & (np.abs(step) > config.convergence_tolerance_m)
            if not unresolved.any():
                break
            step[unresolved] *= 0.5
            candidate[unresolved] = updated[unresolved] + step[unresolved]
        candidate_cost = _node_costs(
            candidate, reference_dem, target_reflectance, photometric_valid,
            pixel_size_x, pixel_size_y, sun_directions, view_directions, config,
        )
        improved = selected & np.isfinite(candidate_cost) & (candidate_cost <= base)
        accepted += int(improved.sum())
        updated[improved] = candidate[improved]
        if improved.any():
            maximum_step = max(maximum_step, float(np.max(np.abs(step[improved]))))

    if config.preserve_vertical_datum:
        offset = float(np.nanmedian((updated - reference_dem)[optimization_mask]))
        if np.isfinite(offset):
            updated[optimization_mask] -= offset
    return updated, maximum_step, accepted / max(attempted, 1)


def _sequential_sweep(
    dem: np.ndarray,
    reference_dem: np.ndarray,
    target_reflectance: np.ndarray,
    photometric_valid: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    config: SafsConfig,
    optimization_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Literal illumination-ordered node-by-node Gauss-Seidel sweep."""
    if dem.size > config.sequential_max_pixels:
        raise ValueError(
            f"Sequential mode is limited to {config.sequential_max_pixels} pixels; "
            "use four_color for a full NAC raster"
        )
    updated = dem.copy()
    sun = normalize_directions(sun_directions)
    if sun.ndim > 1:
        sun = normalize_directions(np.nanmedian(sun.reshape(-1, 3), axis=0))
    row_order = range(1, dem.shape[0] - 1) if sun[1] >= 0 else range(dem.shape[0] - 2, 0, -1)
    column_order = range(1, dem.shape[1] - 1) if sun[0] <= 0 else range(dem.shape[1] - 2, 0, -1)
    epsilon = config.finite_difference_step_m
    maximum_step = 0.0
    accepted = 0
    attempted = 0
    for row in row_order:
        for column in column_order:
            if not optimization_mask[row, column]:
                continue
            attempted += 1
            base = _mean_cost(
                updated, reference_dem, target_reflectance, photometric_valid,
                pixel_size_x, pixel_size_y, sun_directions, view_directions,
                config, optimization_mask,
            )
            updated[row, column] += epsilon
            plus = _mean_cost(
                updated, reference_dem, target_reflectance, photometric_valid,
                pixel_size_x, pixel_size_y, sun_directions, view_directions,
                config, optimization_mask,
            )
            updated[row, column] -= 2.0 * epsilon
            minus = _mean_cost(
                updated, reference_dem, target_reflectance, photometric_valid,
                pixel_size_x, pixel_size_y, sun_directions, view_directions,
                config, optimization_mask,
            )
            updated[row, column] += epsilon
            gradient = (plus - minus) / (2.0 * epsilon)
            curvature = (plus - 2.0 * base + minus) / (epsilon * epsilon)
            step = float(np.clip(
                -gradient / max(curvature, config.newton_damping),
                -config.maximum_height_step_m,
                config.maximum_height_step_m,
            ))
            trial = step
            for _ in range(config.line_search_steps + 1):
                updated[row, column] += trial
                candidate = _mean_cost(
                    updated, reference_dem, target_reflectance, photometric_valid,
                    pixel_size_x, pixel_size_y, sun_directions, view_directions,
                    config, optimization_mask,
                )
                if np.isfinite(candidate) and candidate <= base:
                    accepted += 1
                    maximum_step = max(maximum_step, abs(trial))
                    break
                updated[row, column] -= trial
                trial *= 0.5
            else:
                pass
    if config.preserve_vertical_datum:
        offset = float(np.nanmedian((updated - reference_dem)[optimization_mask]))
        if np.isfinite(offset):
            updated[optimization_mask] -= offset
    return updated, maximum_step, accepted / max(attempted, 1)


def _mean_cost(
    dem: np.ndarray,
    reference_dem: np.ndarray,
    target_reflectance: np.ndarray,
    photometric_valid: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    config: SafsConfig,
    optimization_mask: np.ndarray,
) -> float:
    costs = _node_costs(
        dem, reference_dem, target_reflectance, photometric_valid,
        pixel_size_x, pixel_size_y, sun_directions, view_directions, config,
    )
    valid = optimization_mask & np.isfinite(costs)
    return float(np.mean(costs[valid])) if valid.any() else float("nan")


def solve_safs(
    image: np.ndarray,
    initial_dem: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    config: SafsConfig | None = None,
    valid_mask: np.ndarray | None = None,
    progress: Callable[[dict[str, float]], None] | None = None,
) -> SafsResult:
    """Refine one low-resolution DEM from one image and estimate albedo."""
    settings = config or SafsConfig()
    settings.validate()
    image = np.asarray(image, dtype=np.float64)
    initial_dem = np.asarray(initial_dem, dtype=np.float64)
    if image.ndim != 2 or initial_dem.ndim != 2:
        raise ValueError("image and initial_dem must be two-dimensional")
    if pixel_size_x <= 0.0 or pixel_size_y <= 0.0:
        raise ValueError("Pixel sizes must be positive")
    if not np.isfinite(initial_dem).all():
        raise ValueError("The coarse DEM must be finite over the reconstruction extent")
    # Keep the geometric support separate from the photometric support.  A
    # measured value of zero can be a real cast shadow; it contains no usable
    # reflectance observation, but the DEM must still remain defined there and
    # be propagated by the low-resolution normal constraint.
    if valid_mask is None:
        valid_mask = np.isfinite(image)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(image)
    if valid_mask.sum() < 100:
        raise ValueError("Too few valid image pixels")

    sun_directions = normalize_directions(sun_directions)
    view_directions = normalize_directions(view_directions)
    if sun_directions.ndim > 1 and sun_directions.shape[:2] != image.shape:
        raise ValueError("Per-pixel sun geometry must match the image grid")
    if view_directions.ndim > 1 and view_directions.shape[:2] != image.shape:
        raise ValueError("Per-pixel view geometry must match the image grid")

    shapes = pyramid_shapes(initial_dem.shape, image.shape, settings.maximum_pyramid_levels)
    current_dem = initial_dem.copy()
    history: list[dict[str, float]] = []

    for level_index, shape in enumerate(shapes):
        if current_dem.shape != shape:
            current_dem = resize_bilinear(current_dem, shape)
        reference_dem = current_dem.copy()
        image_level = resize_bilinear(image, shape)
        mask_level = resize_bilinear(valid_mask.astype(np.float64), shape) >= 0.75
        sun_level = resize_directions(sun_directions, shape)
        view_level = resize_directions(view_directions, shape)
        level_pixel_x = pixel_size_x * image.shape[1] / shape[1]
        level_pixel_y = pixel_size_y * image.shape[0] / shape[0]
        finite_positive = mask_level & np.isfinite(image_level) & (image_level > 0.0)
        percentile_cutoff = float(np.percentile(image_level[finite_positive], settings.shadow_percentile))
        image_valid = finite_positive & (
            image_level > max(settings.shadow_threshold_normalized, percentile_cutoff)
        )
        albedo_window = _scaled_window(settings.albedo_window_final_px, shape, image.shape)
        reflectance_window = _scaled_window(
            settings.reflectance_smoothing_final_px, shape, image.shape
        )

        for iteration in range(settings.iterations_per_level):
            node_reflectance, illuminated = render_reflectance(
                current_dem, level_pixel_x, level_pixel_y,
                sun_level, view_level, settings.lunar_lambert_l,
            )
            albedo, exposure = estimate_piecewise_albedo(
                image_level,
                node_reflectance,
                image_valid & illuminated,
                albedo_window,
                settings.albedo_minimum,
                settings.albedo_maximum,
                reflectance_window,
            )
            target_nodes = image_level / np.maximum(exposure * albedo, EPS)
            target_cells = four_corner_mean(target_nodes)
            photometric_valid = four_corner_all(image_valid & illuminated)
            cost_before = _mean_cost(
                current_dem, reference_dem, target_cells, photometric_valid,
                level_pixel_x, level_pixel_y, sun_level, view_level,
                settings, mask_level,
            )
            sweep = _sequential_sweep if settings.sweep_mode == "sequential" else _four_color_sweep
            current_dem, maximum_step, acceptance = sweep(
                current_dem, reference_dem, target_cells, photometric_valid,
                level_pixel_x, level_pixel_y, sun_level, view_level,
                settings, mask_level,
            )
            cost_after = _mean_cost(
                current_dem, reference_dem, target_cells, photometric_valid,
                level_pixel_x, level_pixel_y, sun_level, view_level,
                settings, mask_level,
            )
            record = {
                "level": float(level_index),
                "iteration": float(iteration),
                "rows": float(shape[0]),
                "columns": float(shape[1]),
                "pixel_size_x_m": float(level_pixel_x),
                "pixel_size_y_m": float(level_pixel_y),
                "albedo_window_px": float(albedo_window),
                "reflectance_smoothing_window_px": float(reflectance_window),
                "valid_pixels": float(image_valid.sum()),
                "exposure": float(exposure),
                "cost_before": float(cost_before),
                "cost_after": float(cost_after),
                "maximum_height_step_m": float(maximum_step),
                "accepted_fraction": float(acceptance),
            }
            history.append(record)
            if progress is not None:
                progress(record)
            if maximum_step < settings.convergence_tolerance_m:
                break

    final_reflectance, illuminated = render_reflectance(
        current_dem, pixel_size_x, pixel_size_y,
        sun_directions, view_directions, settings.lunar_lambert_l,
    )
    final_positive = valid_mask & np.isfinite(image) & (image > 0.0)
    cutoff = float(np.percentile(image[final_positive], settings.shadow_percentile))
    final_valid = final_positive & illuminated & (
        image > max(settings.shadow_threshold_normalized, cutoff)
    )
    final_albedo, final_exposure = estimate_piecewise_albedo(
        image,
        final_reflectance,
        final_valid,
        settings.albedo_window_final_px,
        settings.albedo_minimum,
        settings.albedo_maximum,
        settings.reflectance_smoothing_final_px,
    )
    modeled = final_exposure * final_albedo * final_reflectance
    output_dem = current_dem.copy()
    output_dem[~valid_mask] = np.nan
    final_albedo[~valid_mask] = np.nan
    modeled[~valid_mask] = np.nan
    return SafsResult(
        dem=output_dem,
        albedo=final_albedo,
        exposure=final_exposure,
        modeled_intensity=modeled,
        valid_mask=final_valid,
        history=history,
        pyramid_shapes=shapes,
        config=asdict(settings),
    )
