"""Illumination-independent subpixel registration for map-gridded images.

This is the rigid, pairwise branch of Grumpe et al. (2014, section 5.5).
Single-image PHCL produces illumination-independent slope descriptors.  A
multi-level normalized-mutual-information search then estimates translation
and rotation relative to a user-selected reference image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import minimize

from .photoclinometry import estimate_phcl
from .reflectance import HapkeParameters, ReflectanceModel


@dataclass(frozen=True)
class RegistrationResult:
    images: np.ndarray
    sun_directions: np.ndarray
    view_directions: np.ndarray
    descriptors: np.ndarray
    transforms: tuple[dict[str, float], ...]


def _rigid_warp(array: np.ndarray, dy: float, dx: float, angle_degrees: float, order: int) -> np.ndarray:
    """Warp a 2-D grid into the reference grid around the image centre."""
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    # scipy affine_transform maps output coordinates to input coordinates.
    matrix = np.array([[c, s], [-s, c]], dtype=np.float64)
    centre = 0.5 * (np.asarray(array.shape, dtype=np.float64) - 1.0)
    shift = np.array([dy, dx], dtype=np.float64)
    offset = centre - matrix @ (centre + shift)
    return affine_transform(
        np.asarray(array, dtype=np.float64),
        matrix,
        offset=offset,
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=order > 1,
    )


def normalized_mutual_information(reference: np.ndarray, floating: np.ndarray, bins: int = 48) -> float:
    valid = np.isfinite(reference) & np.isfinite(floating)
    if valid.sum() < 64:
        return -np.inf
    x, y = reference[valid], floating[valid]
    xlo, xhi = np.percentile(x, [1.0, 99.0])
    ylo, yhi = np.percentile(y, [1.0, 99.0])
    if not (xhi > xlo and yhi > ylo):
        return -np.inf
    histogram, _, _ = np.histogram2d(
        np.clip(x, xlo, xhi), np.clip(y, ylo, yhi), bins=bins,
        range=((xlo, xhi), (ylo, yhi)),
    )
    probability = histogram / max(float(histogram.sum()), 1.0)
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    joint = probability > 0.0
    hx = -float(np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
    hy = -float(np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
    hxy = -float(np.sum(probability[joint] * np.log(probability[joint])))
    return (hx + hy) / max(hxy, 1.0e-12)


def estimate_rigid_transform(
    reference: np.ndarray,
    floating: np.ndarray,
    *,
    max_shift_px: float = 8.0,
    max_rotation_degrees: float = 0.5,
) -> dict[str, float]:
    """Paper-style Gaussian pyramid (5,3,1,0 px) and subpixel MI search."""
    parameters = np.zeros(3, dtype=np.float64)
    initial_nmi = normalized_mutual_information(reference, floating)
    for sigma in (5.0, 3.0, 1.0, 0.0):
        ref_level = gaussian_filter(reference, sigma, mode="nearest") if sigma else reference
        float_level = gaussian_filter(floating, sigma, mode="nearest") if sigma else floating

        def objective(values: np.ndarray) -> float:
            dy, dx, angle = values
            if abs(dy) > max_shift_px or abs(dx) > max_shift_px or abs(angle) > max_rotation_degrees:
                return 1.0e3 + abs(dy) + abs(dx) + abs(angle)
            warped = _rigid_warp(float_level, dy, dx, angle, 1)
            score = normalized_mutual_information(ref_level, warped)
            return -score if np.isfinite(score) else 1.0e3

        result = minimize(
            objective,
            parameters,
            method="Powell",
            bounds=(
                (-max_shift_px, max_shift_px),
                (-max_shift_px, max_shift_px),
                (-max_rotation_degrees, max_rotation_degrees),
            ),
            options={"xtol": 0.01, "ftol": 1.0e-7, "maxiter": 120},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
    registered = _rigid_warp(floating, *parameters, order=1)
    return {
        "dy_px": float(parameters[0]),
        "dx_px": float(parameters[1]),
        "rotation_degrees": float(parameters[2]),
        "nmi_before": float(initial_nmi),
        "nmi_after": float(normalized_mutual_information(reference, registered)),
    }


def register_by_single_image_phcl(
    images: np.ndarray,
    sun_directions: np.ndarray,
    view_directions: np.ndarray,
    reference_dem: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    parameters: HapkeParameters,
    reflectance_model: ReflectanceModel = "amsa",
    reference_index: int = 0,
    phcl_iterations: int = 50,
) -> RegistrationResult:
    """Estimate one illumination-independent slope descriptor per image."""
    observed = np.asarray(images, dtype=np.float64)
    sun = np.asarray(sun_directions, dtype=np.float64)
    view = np.asarray(view_directions, dtype=np.float64)
    image_count = observed.shape[0]
    descriptors = []
    for index in range(image_count):
        single = estimate_phcl(
            observed[index : index + 1],
            sun[index : index + 1],
            view[index : index + 1],
            reference_dem,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            parameters=parameters,
            reflectance_model=reflectance_model,
            dem_weight=8.0e-4,
            dem_sigma_px=15.0,
            albedo_sigma_px=21.0,
            max_iterations=phcl_iterations,
            max_rejections=30,
            min_observations=1,
        )
        # Gradient orientation as two bounded channels; magnitude alone loses
        # too much information on weakly sloped terrain.
        descriptor = np.arctan2(single.q, single.p) + np.hypot(single.p, single.q)
        descriptors.append(descriptor)
    descriptors_array = np.asarray(descriptors)
    output_images = np.full_like(observed, np.nan)
    output_sun = np.full_like(sun, np.nan)
    output_view = np.full_like(view, np.nan)
    transforms: list[dict[str, float]] = []
    reference_descriptor = descriptors_array[reference_index]
    for index in range(image_count):
        if index == reference_index:
            transform = {
                "dy_px": 0.0,
                "dx_px": 0.0,
                "rotation_degrees": 0.0,
                "nmi_before": normalized_mutual_information(reference_descriptor, descriptors_array[index]),
                "nmi_after": normalized_mutual_information(reference_descriptor, descriptors_array[index]),
            }
        else:
            transform = estimate_rigid_transform(reference_descriptor, descriptors_array[index])
        transforms.append(transform)
        values = (transform["dy_px"], transform["dx_px"], transform["rotation_degrees"])
        output_images[index] = _rigid_warp(observed[index], *values, order=3)
        for component in range(3):
            output_sun[index, ..., component] = _rigid_warp(sun[index, ..., component], *values, order=1)
            output_view[index, ..., component] = _rigid_warp(view[index, ..., component], *values, order=1)
        sun_norm = np.linalg.norm(output_sun[index], axis=-1, keepdims=True)
        view_norm = np.linalg.norm(output_view[index], axis=-1, keepdims=True)
        output_sun[index] /= np.where(sun_norm > 0.0, sun_norm, np.nan)
        output_view[index] /= np.where(view_norm > 0.0, view_norm, np.nan)
    return RegistrationResult(
        images=output_images,
        sun_directions=output_sun,
        view_directions=output_view,
        descriptors=descriptors_array,
        transforms=tuple(transforms),
    )
