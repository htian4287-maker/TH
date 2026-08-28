import numpy as np

from safs_method import SafsConfig, pyramid_shapes, solve_safs
from safs_method.model import resize_bilinear
from safs_method.synthetic import make_synthetic_case


def test_pyramid_starts_on_actual_paper_coarse_grid():
    assert pyramid_shapes((12, 11), (1382, 1262), 9) == [
        (12, 11), (24, 22), (48, 44), (96, 88),
        (192, 176), (384, 352), (768, 704), (1382, 1262),
    ]


def test_synthetic_reconstruction_beats_interpolation():
    image, coarse, truth, _, sun, view = make_synthetic_case(48)
    baseline = resize_bilinear(coarse, truth.shape)
    baseline_rmse = float(np.sqrt(np.mean((baseline - truth) ** 2)))
    result = solve_safs(
        image, coarse, 1.0, 1.0, sun, view,
        SafsConfig(iterations_per_level=3, maximum_pyramid_levels=7),
    )
    valid = np.isfinite(result.dem)
    reconstructed_rmse = float(np.sqrt(np.mean((result.dem[valid] - truth[valid]) ** 2)))
    assert result.dem.shape == image.shape
    assert result.albedo.shape == image.shape
    assert reconstructed_rmse < baseline_rmse
    assert len(result.history) > 0


def test_literal_sequential_and_four_color_both_reduce_cost():
    image, coarse, _, _, sun, view = make_synthetic_case(24)
    for mode in ("sequential", "four_color"):
        result = solve_safs(
            image, coarse, 1.0, 1.0, sun, view,
            SafsConfig(
                iterations_per_level=1,
                maximum_pyramid_levels=4,
                sweep_mode=mode,
                sequential_max_pixels=10000,
            ),
        )
        assert all(row["cost_after"] <= row["cost_before"] * 1.001 for row in result.history)
