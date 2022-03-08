import numpy as np
from .. import Vec2


def _assert_valid(distance_to_z_axis, curvature_radius):
    assert np.abs(curvature_radius) >= distance_to_z_axis
    assert distance_to_z_axis >= 0.0


def surface_height(x, curvature_radius):
    distance_to_y_axis = x
    _assert_valid(distance_to_y_axis, curvature_radius)
    sig = np.sign(curvature_radius)
    h = np.abs(curvature_radius) - np.sqrt(
        curvature_radius ** 2 - distance_to_y_axis ** 2
    )
    return sig * h


def surface_normal(x, curvature_radius):
    distance_to_y_axis = x
    _assert_valid(distance_to_y_axis, curvature_radius)
    center_of_sphere = Vec2.Vec2(x=0.0, y=curvature_radius)
    point_on_sphere = Vec2.Vec2(x=x, y=surface_height(x, curvature_radius))
    diff = Vec2.subtract(center_of_sphere, point_on_sphere)
    scale = np.sign(curvature_radius) / Vec2.norm(diff)
    return Vec2.multiply(diff, scale)


def intersection_ray_parameters(ray, curvature_radius):
    r = curvature_radius
    xd = ray.direction.x
    yd = ray.direction.y
    xs = ray.support.x
    ys = ray.support.y

    div = xd ** 2 + yd ** 2
    p = (2 * xs * xd + 2 * ys * yd) / div
    q = (xs ** 2 + ys ** 2 - r ** 2) / div
    inner_sqrt = (p / 2) ** 2 - q
    _sqrt = np.sqrt(inner_sqrt)

    a_plus = -p / 2 + _sqrt
    a_minus = -p / 2 - _sqrt
    return a_plus, a_minus, inner_sqrt


def intersection_ray(ray, curvature_radius):
    ap, am, insqrt = intersection_ray_parameters(
        ray=ray,
        curvature_radius=curvature_radius
    )

    if insqrt >= 0:
        if ap > 0.0 or am > 0.0:
            if ap < 0.0:
                # is am
            elif am < 0.0:
                # is ap
            else:
                if ap < am:
                    # is ap
                else:
                    # is am
        else:
            return None
    else:
        return None