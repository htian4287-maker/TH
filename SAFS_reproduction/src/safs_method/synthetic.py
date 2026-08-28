"""Synthetic lunar-like data used for regression tests and parameter checks."""

from __future__ import annotations

import numpy as np

from .model import direction_from_azimuth_elevation, render_reflectance, resize_bilinear


def make_synthetic_case(size: int, seed: int = 7):
    y, x = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    radius = np.sqrt((x + 0.05) ** 2 + (y - 0.03) ** 2)
    crater = -5.0 * np.exp(-(radius / 0.24) ** 4)
    rim = 1.2 * np.exp(-((radius - 0.30) / 0.055) ** 2)
    boulder_1 = 0.9 * np.exp(-((x - 0.38) ** 2 + (y + 0.15) ** 2) / 0.003)
    boulder_2 = 0.6 * np.exp(-((x + 0.30) ** 2 + (y + 0.38) ** 2) / 0.002)
    broad_shape = 1.3 * x - 0.7 * y + 0.4 * x * y
    truth_dem = broad_shape + crater + rim + boulder_1 + boulder_2

    truth_albedo = np.ones_like(truth_dem)
    truth_albedo += 0.20 * np.exp(-((x + 0.42) ** 2 + (y - 0.35) ** 2) / 0.06)
    truth_albedo -= 0.12 * np.exp(-((x - 0.25) ** 2 + (y - 0.45) ** 2) / 0.04)

    sun = direction_from_azimuth_elevation(135.0, 28.0)
    view = direction_from_azimuth_elevation(0.0, 90.0)
    reflectance, illuminated = render_reflectance(
        truth_dem, 1.0, 1.0, sun, view, 0.55
    )
    rng = np.random.default_rng(seed)
    image = 1.4 * truth_albedo * reflectance
    image += rng.normal(0.0, 0.003, image.shape)
    image[~illuminated] = 0.0
    positive = image > 0.0
    image /= np.median(image[positive])

    coarse_size = max(6, size // 12)
    initial_dem = resize_bilinear(truth_dem, (coarse_size, coarse_size))
    return image, initial_dem, truth_dem, truth_albedo, sun, view

