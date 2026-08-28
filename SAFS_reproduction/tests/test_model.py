import numpy as np

from safs_method.model import (
    cell_slopes,
    direction_from_azimuth_zenith,
    render_reflectance,
)


def test_equation_2_recovers_planar_slopes_on_north_up_grid():
    rows, columns = np.indices((13, 17))
    pixel_x, pixel_y = 2.0, 3.0
    east_coordinate = columns * pixel_x
    north_coordinate = -rows * pixel_y
    dem = 7.0 + 0.12 * east_coordinate - 0.08 * north_coordinate
    p, q = cell_slopes(dem, pixel_x, pixel_y)
    np.testing.assert_allclose(p, 0.12, atol=1e-12)
    np.testing.assert_allclose(q, -0.08, atol=1e-12)


def test_per_pixel_and_constant_geometry_agree():
    dem = np.zeros((16, 18), dtype=np.float64)
    sun = direction_from_azimuth_zenith(120.0, 55.0)
    view = direction_from_azimuth_zenith(180.0, 10.0)
    constant, valid_constant = render_reflectance(dem, 1.0, 1.0, sun, view, 0.55)
    sun_field = np.broadcast_to(sun, (*dem.shape, 3)).copy()
    view_field = np.broadcast_to(view, (*dem.shape, 3)).copy()
    field, valid_field = render_reflectance(dem, 1.0, 1.0, sun_field, view_field, 0.55)
    np.testing.assert_allclose(field, constant, atol=1e-12)
    np.testing.assert_array_equal(valid_field, valid_constant)
