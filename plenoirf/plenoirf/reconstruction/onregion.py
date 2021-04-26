import numpy as np

def make_polygon(onregion, num_steps=1000):
    return _make_ellipse_polygon(
        center_x=onregion["ellipse_center_cx"],
        center_y=onregion["ellipse_center_cy"],
        mayor_radius=onregion["ellipse_mayor_radius"],
        minor_radius=onregion["ellipse_minor_radius"],
        main_axis_azimuth=onregion["ellipse_main_axis_azimuth"],
        num_steps=num_steps
    )


def _make_ellipse_polygon(
    center_x,
    center_y,
    mayor_radius,
    minor_radius,
    main_axis_azimuth,
    num_steps=1000
):
    # polar coordinates
    theta = np.linspace(0, 2 * np.pi, num_steps)
    radius = (mayor_radius*minor_radius) / np.sqrt(
        (minor_radius*np.cos(theta))**2 +
        (mayor_radius*np.sin(theta))**2
    )

    xs = center_x + np.cos(theta + main_axis_azimuth) * radius
    ys = center_y + np.sin(theta + main_axis_azimuth) * radius
    return xs, ys



def ellipse_mayor_minor_ratio(
    reco_core_radius,
    core_radius_uncertainty_doubling,
):
    return 1.0 + reco_core_radius/core_radius_uncertainty_doubling


def estimate_onregion(
    reco_cx,
    reco_cy,
    reco_main_axis_azimuth,
    reco_num_photons,
    reco_core_radius,
    core_radius_uncertainty_doubling,
    opening_angle_vs_reco_num_photons,
):
    ellipse_opening_angle = np.interp(
        x=reco_num_photons,
        xp=opening_angle_vs_reco_num_photons["num_photons_pe"],
        fp=np.deg2rad(opening_angle_vs_reco_num_photons["opening_angle_deg"])
    )

    ellipse_ratio = ellipse_mayor_minor_ratio(
        reco_core_radius=reco_core_radius,
        core_radius_uncertainty_doubling=core_radius_uncertainty_doubling,
    )

    # constant solid angle: A = pi*(ratio*a)*(1/ratio*b)
    ellipse_mayor_radius = ellipse_opening_angle*ellipse_ratio
    ellipse_minor_radius = ellipse_opening_angle/ellipse_ratio
    ellipse_solid_angle = np.pi * ellipse_mayor_radius * ellipse_minor_radius

    return {
        "ellipse_center_cx": float(reco_cx),
        "ellipse_center_cy": float(reco_cy),
        "ellipse_main_axis_azimuth": float(reco_main_axis_azimuth),
        "ellipse_mayor_radius": ellipse_mayor_radius,
        "ellipse_minor_radius": ellipse_minor_radius,
        "ellipse_solid_angle": ellipse_solid_angle
    }


def is_direction_inside(cx, cy, onregion):
    return _is_point_inside_ellipse(
        point_x=cx,
        point_y=cy,
        ellipse_center_x=onregion["ellipse_center_cx"],
        ellipse_center_y=onregion["ellipse_center_cy"],
        ellipse_mayor_radius=onregion["ellipse_mayor_radius"],
        ellipse_minor_radius=onregion["ellipse_minor_radius"],
        ellipse_main_axis_azimuth=onregion["ellipse_main_axis_azimuth"],
    )



def _is_point_inside_ellipse(
    point_x,
    point_y,
    ellipse_center_x,
    ellipse_center_y,
    ellipse_mayor_radius,
    ellipse_minor_radius,
    ellipse_main_axis_azimuth,
):
    tpx = point_x - ellipse_center_x
    tpy = point_y - ellipse_center_y

    a = -1.0*ellipse_main_axis_azimuth
    rpx = tpx*np.cos(a) -tpy*np.sin(a)
    rpy = tpx*np.sin(a) +tpy*np.cos(a)

    return _is_point_inside_centered_ellipse_with_mayor_axis_on_x(
        point_x=rpx,
        point_y=rpy,
        ellipse_mayor_along_x_radius=ellipse_mayor_radius,
        ellipse_minor_along_y_radius=ellipse_minor_radius,
    )


def _is_point_inside_centered_ellipse_with_mayor_axis_on_x(
    point_x,
    point_y,
    ellipse_mayor_along_x_radius,
    ellipse_minor_along_y_radius,
):
    return (
        (point_x**2) / (ellipse_mayor_along_x_radius**2) +
        (point_y**2) / (ellipse_minor_along_y_radius**2)
    ) <= 1.0
