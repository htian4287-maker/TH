from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from grumpe_method.integration import forward_gradient, integrate_gradients


def test_exact_gradient_recovery_up_to_datum() -> None:
    yy, xx = np.mgrid[0:32, 0:40]
    truth = 0.1 * xx - 0.07 * yy + np.sin(xx / 5.0)
    p, q = forward_gradient(truth, 2.0, 2.0)
    result = integrate_gradients(
        p, q, np.zeros_like(truth), pixel_size_x=2.0, pixel_size_y=2.0,
        depth_weight=0.0, lowpass_sigma_px=0.0,
    )
    actual = result.dem - np.mean(result.dem)
    expected = truth - np.mean(truth)
    assert result.converged
    assert np.sqrt(np.mean((actual - expected) ** 2)) < 1.0e-4


def test_lowpass_constraint_retains_missing_detail_better_than_raw_constraint() -> None:
    yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
    truth = 4.0 * xx - 2.0 * yy - 5.0 * np.exp(-((xx + 0.1) ** 2 + yy**2) / 0.02)
    prior = gaussian_filter(truth, 6.0, mode="reflect")
    p, q = forward_gradient(truth, 2.0, 2.0)
    raw = integrate_gradients(
        p, q, prior, pixel_size_x=2.0, pixel_size_y=2.0,
        depth_weight=2.0, lowpass_sigma_px=0.0,
    ).dem
    lowpass = integrate_gradients(
        p, q, prior, pixel_size_x=2.0, pixel_size_y=2.0,
        depth_weight=2.0, lowpass_sigma_px=6.0,
    ).dem
    raw_rmse = np.sqrt(np.mean((raw - truth) ** 2))
    lowpass_rmse = np.sqrt(np.mean((lowpass - truth) ** 2))
    assert lowpass_rmse < raw_rmse

