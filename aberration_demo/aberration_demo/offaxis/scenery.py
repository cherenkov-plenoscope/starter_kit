import numpy as np
from .. import portal


def make_mirror_davies_cotton(focal_length, outer_radius):
    return {
        "type": "SegmentedReflector",
        "pos": [0, 0, 0],
        "rot": [0, 0, 1.570796],
        "focal_length": focal_length,
        "max_outer_aperture_radius": outer_radius,
        "min_inner_aperture_radius": 0.0,
        "outer_aperture_shape_hex": True,
        "DaviesCotton_over_parabolic_mixing_factor": 1.0,
        "facet_inner_hex_radius": 0.75,
        "gap_between_facets": 0.025,
        "name": "reflector",
        "surface": {"outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": [],
    }


def make_mirror_parabola_segmented(focal_length, outer_radius):
    return {
        "type": "SegmentedReflector",
        "pos": [0, 0, 0],
        "rot": [0, 0, 1.570796],
        "focal_length": focal_length,
        "max_outer_aperture_radius": outer_radius,
        "min_inner_aperture_radius": 0.0,
        "outer_aperture_shape_hex": True,
        "DaviesCotton_over_parabolic_mixing_factor": 0.0,
        "facet_inner_hex_radius": 0.75,
        "gap_between_facets": 0.025,
        "name": "reflector",
        "surface": {"outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": [],
    }


def make_mirror_spherical_monolith(focal_length, outer_radius):
    return {
        "type": "SphereCapWithHexagonalBound",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0],
        "curvature_radius": 2.0 * focal_length,
        "outer_radius": outer_radius,
        "name": "reflector",
        "surface": {"outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": [],
    }


MIRRORS = {
    "sphere_monolith": make_mirror_spherical_monolith,
    "davies_cotton": make_mirror_davies_cotton,
    "parabola_segmented": make_mirror_parabola_segmented,
}


def make_plenoscope_scenery_for_merlict(
    mirror_key, num_paxel_on_diagonal, config, off_axis_angles_deg,
):
    mirror_frame = MIRRORS[mirror_key](
        focal_length=config["mirror"]["focal_length"],
        outer_radius=config["mirror"]["outer_radius"],
    )

    # ------------------------------------------
    alpha = np.deg2rad(off_axis_angles_deg)
    f = config["mirror"]["focal_length"]

    posx = f * np.sin(alpha)
    posy = 0.0
    posz = f * np.cos(alpha)

    sensor_frame = {
        "type": "LightFieldSensor",
        "name": "light_field_sensor",
        "pos": [posx, posy, posz],
        "rot": [0, -alpha, 0],
        "expected_imaging_system_focal_length": config["mirror"][
            "focal_length"
        ],
        "expected_imaging_system_aperture_radius": config["mirror"][
            "inner_radius"
        ],
        "max_FoV_diameter_deg": 2.0 * config["sensor"]["fov_radius_deg"],
        "hex_pixel_FoV_flat2flat_deg": config["sensor"][
            "hex_pixel_fov_flat2flat_deg"
        ],
        "num_paxel_on_pixel_diagonal": num_paxel_on_diagonal,
        "housing_overhead": config["sensor"]["housing_overhead"],
        "lens_refraction_vs_wavelength": "lens_refraction_vs_wavelength",
        "bin_reflection_vs_wavelength": "mirror_reflectivity_vs_wavelength",
        "children": [],
    }

    scn = {
        "functions": [
            portal.MIRROR_REFLECTIVITY_VS_WAVELENGTH,
            portal.LENS_REFRACTION_VS_WAVELENGTH,
        ],
        "colors": [],
        "children": [
            {
                "type": "Frame",
                "name": "Portal",
                "pos": [0, 0, 0],
                "rot": [0, -alpha, 0],
                "children": [mirror_frame, sensor_frame],
            }
        ],
    }

    return scn
