"""Controlled synthetic terrain and Hapke observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import direction_from_azimuth_zenith
from .photoclinometry import reference_slopes, render_hapke_stack
from .reflectance import HapkeParameters, ReflectanceModel


@dataclass(frozen=True)
class SyntheticCase:
    truth_dem: np.ndarray
    initial_dem: np.ndarray
    truth_p: np.ndarray
    truth_q: np.ndarray
    truth_w: np.ndarray
    images: np.ndarray
    sun_directions: np.ndarray
    view_directions: np.ndarray
    pixel_size_m: float


def make_synthetic_case(
    *,
    size: int = 72,
    pixel_size_m: float = 2.0,
    prior_sigma_px: float = 7.0,
    noise_std: float = 8.0e-5,
    seed: int = 42,
    parameters: HapkeParameters | None = None,
    reflectance_model: ReflectanceModel = "imsa",
) -> SyntheticCase:
    if size < 32:
        raise ValueError("Synthetic grid must be at least 32x32")
    parameters = parameters or HapkeParameters()
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    radius = np.hypot(xx + 0.18, yy - 0.08)
    crater = -7.0 * np.exp(-(radius / 0.23) ** 4)
    rim = 1.7 * np.exp(-((radius - 0.29) / 0.045) ** 2)
    small_crater = -2.0 * np.exp(-((xx - 0.42) ** 2 + (yy + 0.28) ** 2) / 0.012)
    ridge = 1.2 * np.exp(-((yy + 0.42 - 0.18 * xx) / 0.05) ** 2)
    broad = 2.2 * xx - 1.6 * yy + 0.5 * xx * yy
    truth = broad + crater + rim + small_crater + ridge
    initial = gaussian_filter(truth, prior_sigma_px, mode="reflect")
    truth_p, truth_q = reference_slopes(truth, pixel_size_m, pixel_size_m)
    truth_w = 0.42 + 0.045 * np.exp(-((xx + 0.35) ** 2 + (yy - 0.35) ** 2) / 0.08)
    truth_w -= 0.035 * (radius < 0.18)
    truth_w = np.clip(truth_w, 0.25, 0.65)

    sun_azimuths = np.array([35.0, 92.0, 148.0, 218.0, 278.0, 332.0])
    sun_zeniths = np.array([50.0, 62.0, 55.0, 58.0, 66.0, 52.0])
    view_azimuths = np.array([15.0, 70.0, 130.0, 205.0, 265.0, 320.0])
    view_zeniths = np.array([4.0, 7.0, 5.0, 6.0, 8.0, 5.0])
    suns = direction_from_azimuth_zenith(sun_azimuths, sun_zeniths)
    views = direction_from_azimuth_zenith(view_azimuths, view_zeniths)
    images = render_hapke_stack(
        truth_p, truth_q, truth_w, suns, views, parameters, reflectance_model
    )
    rng = np.random.default_rng(seed)
    images = images + rng.normal(0.0, noise_std, size=images.shape)
    images = np.where(np.isfinite(images), np.maximum(images, 1.0e-9), np.nan)
    return SyntheticCase(
        truth_dem=truth,
        initial_dem=initial,
        truth_p=truth_p,
        truth_q=truth_q,
        truth_w=truth_w,
        images=images,
        sun_directions=suns,
        view_directions=views,
        pixel_size_m=float(pixel_size_m),
    )
