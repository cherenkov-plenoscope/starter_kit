import numpy as np


CONFIG = {}
CONFIG["focal_length"] = 106.5
CONFIG["max_outer_aperture_radius"] = 41.0
CONFIG["min_inner_aperture_radius"] = 0.0
CONFIG["outer_aperture_shape_hex"] = True
CONFIG["facet_inner_hex_radius"] = 0.75
CONFIG["gap_between_facets"] = 0.025
CONFIG["deformation_polynom"] = [[0,0], [0,0], [1e-3, -1e-4]]


def z_parabola(distance_to_z_axis, focal_length):
    return 1.0 / (4.0*focal_length) * distance_to_z_axis ** 2


def surface_z(x, y, focal_length, deformation_polynom):
    _z_parabola = z_parabola(
        distance_to_z_axis=np.hypot(x, y),
        focal_length=focal_length,
    )
    _z_deformation = np.polynomial.polynomial.polyval2d(
        x=x,
        y=y,
        c=deformation_polynom,
    )
    return _z_parabola + _z_deformation


def surface_normal(x, y, focal_length, deformation_polynom, delta=1e-6):
    """
    surface-normal is: ( -dz/dx , -dz/dy , 1 )
    """
    xp = x + delta
    xm = x - delta
    yp = y + delta
    ym = y - delta
    z_xp = surface_z(xp, y, focal_length, deformation_polynom)
    z_xm = surface_z(xm, y, focal_length, deformation_polynom)
    z_yp = surface_z(x, yp, focal_length, deformation_polynom)
    z_ym = surface_z(x, ym, focal_length, deformation_polynom)
    dzdx = (z_xp - z_xm) / (2 * delta)
    dzdy = (z_yp - z_ym) / (2 * delta)
    normal = [-dzdx, -dzdy, 1.0]
    return normal / np.linalg.norm(normal)


def make_rot_axis_and_angle(normal):
    UNIT_Z = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(UNIT_Z, normal)
    angle_to_unit_z = np.arccos(np.dot(UNIT_Z, normal))
    return rot_axis, angle_to_unit_z


def make_facets(
    config,
    reflection_vs_wavelength='facet_reflection_vs_wavelength',
    color='facet_color',
):
    facet_spacing = (
        config["facet_inner_hex_radius"] * 2.0 + config["gap_between_facets"]
    )
    max_outer_radius_to_put_facet_center = (
        config["max_outer_aperture_radius"] - facet_spacing / 2.0
    )
    min_inner_radius_to_put_facet_center = (
        config["min_inner_aperture_radius"] + facet_spacing / 2.0
    )
    UNIT_X = np.array([1, 0, 0])
    UNIT_Y = np.array([0, 1, 0])

    HEX_B = UNIT_Y * facet_spacing
    HEX_A = UNIT_Y * 0.5 + UNIT_X * (np.sqrt(3.) / 2.) * facet_spacing

    UNIT_U = UNIT_X
    UNIT_V = UNIT_Y * np.sin(2./3.*np.pi) + UNIT_X * np.cos(2./3.*np.pi)
    UNIT_W = UNIT_Y * -np.sin(2./3.*np.pi) + UNIT_X * np.cos(2./3.*np.pi)

    R = (np.sqrt(3.0)/2.0) * max_outer_radius_to_put_facet_center
    N = 2.0 * np.ceil(config["max_outer_aperture_radius"] / facet_spacing)

    facets = []
    for a in np.arange(-N, N+1):
        for b in np.arange(-N, N+1):
            facet_center = HEX_A * a + HEX_B * b

            u = np.dot(UNIT_U, facet_center)
            v = np.dot(UNIT_V, facet_center)
            w = np.dot(UNIT_W, facet_center)

            inside_outer_hexagon = (
                u < R and u > -R and
                v < R and v > -R and
                w < R and w > -R
            )

            outside_inner_disc = (
                np.hypot(facet_center[0], facet_center[1]) >
                min_inner_radius_to_put_facet_center
            )

            if (inside_outer_hexagon and outside_inner_disc):

                facet_center[2] = surface_z(
                    x=facet_center[0],
                    y=facet_center[1],
                    focal_length=config["focal_length"],
                    deformation_polynom=config["deformation_polynom"],
                )

                facet = {}
                facet["type"] = "SphereCapWithHexagonalBound"
                facet["name"] = "facet_{:06d}".format(len(facets))
                facet["pos"] = facet_center
                facet_normal = surface_normal(
                    x=facet_center[0],
                    y=facet_center[1],
                    focal_length=config["focal_length"],
                    deformation_polynom=config["deformation_polynom"],
                    delta=1e-6 * config["focal_length"],
                )
                axis, angle = make_rot_axis_and_angle(normal=facet_normal)
                facet["rot_axis"] = axis
                facet["rot_angle"] = angle

                facet["outer_radius"] = (2 / np.sqrt(3)) * config["facet_inner_hex_radius"]
                facet["curvature_radius"] = 2.0 * config["focal_length"]
                facet["surface"] = {
                    "outer_color": color,
                    "outer_reflection": reflection_vs_wavelength,
                }
                facet["children"]: []
                facets.append(facet)
    return facets
