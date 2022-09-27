import plenoirf
import numpy as np


def test_cone_zero_solid_angle():
    sa = plenoirf.utils.cone_solid_angle(cone_radial_opening_angle_rad=0.0)
    assert sa == 0.0


def test_cone_zero_opening_angle():
    oa = plenoirf.utils.cone_radial_opening_angle(solid_angle=0.0)
    assert oa == 0


def test_cone_conversion_forth_and_back():
    for opening_angle in np.linspace(0, np.pi, 137):
        solid_angle = plenoirf.utils.cone_solid_angle(opening_angle)
        opening_angle_back = plenoirf.utils.cone_radial_opening_angle(solid_angle)

        np.testing.assert_approx_equal(
            actual=opening_angle_back,
            desired=opening_angle,
            significant=7,
        )