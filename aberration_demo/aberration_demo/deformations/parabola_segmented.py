import numpy as np
import copy
from . import deformation_map
from .. import portal


MIRROR = copy.deepcopy(portal.MIRROR)


def parabola_z(distance_to_z_axis, focal_length):
    return 1.0 / (4.0 * focal_length) * distance_to_z_axis ** 2


def mirror_surface_z(x, y, focal_length, deformation):
    """
    Returns the surface height (z-axis) of the mirror.
    The facets are mounted on this surface.
    When the surface deforms, the facets rotate and translate accordingly.
    """
    _z_parabola = parabola_z(
        distance_to_z_axis=np.hypot(x, y), focal_length=focal_length,
    )
    _z_deformation = deformation_map.evaluate(
        deformation_map=deformation, x_m=x, y_m=y,
    )
    return _z_parabola + _z_deformation


def mirror_surface_normal(x, y, focal_length, deformation, delta):
    """
    Returns the mirror's surface-normal: ( -dz/dx , -dz/dy , 1 ) at position
    (x, y). Computed by numeri means.

    Parameters
    ----------
    x : float / m
        The x-coordinate.
    y : float / m
        The y-coordinate.
    focal_length : float m
        Mirror's focal-length
    deformation : dict
        The deformation of the mirror along the z-axis. This is the
        deviation from the targeted geometry.
    delta : float / m
        Step-length in x, and y to sample the neighborhood of (x, y) in order
        to compute the surface's normal.
    """
    xp = x + delta
    xm = x - delta
    yp = y + delta
    ym = y - delta
    z_xp = mirror_surface_z(xp, y, focal_length, deformation)
    z_xm = mirror_surface_z(xm, y, focal_length, deformation)
    z_yp = mirror_surface_z(x, yp, focal_length, deformation)
    z_ym = mirror_surface_z(x, ym, focal_length, deformation)
    dzdx = (z_xp - z_xm) / (2 * delta)
    dzdy = (z_yp - z_ym) / (2 * delta)
    normal = [-dzdx, -dzdy, 1.0]
    return normal / np.linalg.norm(normal)


def angle_between(a, b):
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_rot_axis_and_angle(normal):
    UNIT_Z = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(UNIT_Z, normal)
    angle_to_unit_z = np.arccos(np.dot(UNIT_Z, normal))
    return rot_axis, angle_to_unit_z


UNIT_X = np.array([1, 0, 0])
UNIT_Y = np.array([0, 1, 0])

HEX_B = UNIT_Y
HEX_A = UNIT_Y * 0.5 + UNIT_X * (np.sqrt(3.0) / 2.0)

UNIT_U = UNIT_X
UNIT_V = UNIT_Y * np.sin(2.0 / 3.0 * np.pi) + UNIT_X * np.cos(
    2.0 / 3.0 * np.pi
)
UNIT_W = UNIT_Y * -np.sin(2.0 / 3.0 * np.pi) + UNIT_X * np.cos(
    2.0 / 3.0 * np.pi
)


def is_inside_hexagon(position, hexagon_inner_radius):
    R = hexagon_inner_radius
    u = np.dot(UNIT_U, position)
    v = np.dot(UNIT_V, position)
    w = np.dot(UNIT_W, position)
    inside_outer_hexagon = (
        u < R and u > -R and v < R and v > -R and w < R and w > -R
    )
    return inside_outer_hexagon


def make_facets(
    mirror_config,
    mirror_deformation,
    reflection_vs_wavelength="mirror_reflectivity_vs_wavelength",
    color="facet_color",
):
    """
    Returns a list of facet-dicts for the merlict-raytracer which form
    a segmented mirror with specific deformations.

    Parameters
    ----------
    mirror_config : dict
        Describes the targeted geometry of the mirror.
    mirror_deformation : dict
        Describes the deformations of the mirror, i.e. the deviations from the
        targeted geometry.
    """
    mcfg = mirror_config

    facet_spacing = (
        mcfg["facet_inner_hex_radius"] * 2.0 + mcfg["gap_between_facets"]
    )
    outer_radius_to_put_facet_center = (
        mcfg["max_outer_aperture_radius"] - facet_spacing / 2.0
    )

    hexagon_inner_radius = (
        np.sqrt(3.0) / 2.0
    ) * outer_radius_to_put_facet_center

    MIN_INNER_RADIUS_TO_PUT_FACET_CENTER = (
        mcfg["min_inner_aperture_radius"] + facet_spacing / 2.0
    )

    N = 2.0 * np.ceil(mcfg["max_outer_aperture_radius"] / facet_spacing)

    facets = []
    for a in np.arange(-N, N + 1):
        for b in np.arange(-N, N + 1):
            facet_center = (HEX_A * a + HEX_B * b) * facet_spacing

            inside_outer_hexagon = is_inside_hexagon(
                position=facet_center,
                hexagon_inner_radius=hexagon_inner_radius,
            )

            outside_inner_disc = (
                np.hypot(facet_center[0], facet_center[1],)
                > MIN_INNER_RADIUS_TO_PUT_FACET_CENTER
            )

            if inside_outer_hexagon and outside_inner_disc:
                facet_center[2] = mirror_surface_z(
                    x=facet_center[0],
                    y=facet_center[1],
                    focal_length=mcfg["focal_length"],
                    deformation=mirror_deformation,
                )

                facet = {}
                facet["type"] = "SphereCapWithHexagonalBound"
                facet["name"] = "facet_{:06d}".format(len(facets))
                facet["pos"] = facet_center
                facet_normal = mirror_surface_normal(
                    x=facet_center[0],
                    y=facet_center[1],
                    focal_length=mcfg["focal_length"],
                    deformation=mirror_deformation,
                    delta=0.5 * mcfg["facet_inner_hex_radius"],
                )
                axis, angle = make_rot_axis_and_angle(normal=facet_normal)
                facet["rot_axis"] = axis
                facet["rot_angle"] = angle

                facet["outer_radius"] = (2 / np.sqrt(3)) * mcfg[
                    "facet_inner_hex_radius"
                ]
                facet["curvature_radius"] = 2.0 * mcfg["focal_length"]
                facet["surface"] = {
                    "outer_color": color,
                    "outer_reflection": reflection_vs_wavelength,
                }
                facet["children"] = []
                facets.append(facet)
    return facets
