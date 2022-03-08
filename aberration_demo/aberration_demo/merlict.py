import numpy as np
import os

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
        "surface": {
            "outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": []
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
        "surface": {
            "outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": []
    }


def make_mirror_spherical_monolith(focal_length, outer_radius):
    return {
        "type": "SphereCapWithHexagonalBound",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0],
        "curvature_radius": 2.0 * focal_length,
        "outer_radius": outer_radius,
        "name": "reflector",
        "surface": {
            "outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": []
    }


MIRRORS = {
    "sphere_monolith": make_mirror_spherical_monolith,
    "davies_cotton": make_mirror_davies_cotton,
    "parabola_segmented": make_mirror_parabola_segmented,
}

def make_plenoscope_scenery_for_merlict(
    mirror_key,
    num_paxel_on_diagonal,
    cfg,
):
    mirror = MIRRORS[mirror_key](
        focal_length=cfg["mirror"]["focal_length"],
        outer_radius=cfg["mirror"]["outer_radius"],
    )

    # ------------------------------------------
    sensor_part = {
        "type": "LightFieldSensor",
        "name": "light_field_sensor",
        "pos": [0, 0, cfg["mirror"]["focal_length"]],
        "rot": [0, 0, 0],
        "expected_imaging_system_focal_length": cfg["mirror"]["focal_length"],
        "expected_imaging_system_aperture_radius": cfg["mirror"]["inner_radius"],
        "max_FoV_diameter_deg": 2.0 * cfg["sensor"]["fov_radius_deg"],
        "hex_pixel_FoV_flat2flat_deg": cfg["sensor"]["hex_pixel_fov_flat2flat_deg"],
        "num_paxel_on_pixel_diagonal": num_paxel_on_diagonal,
        "housing_overhead": cfg["sensor"]["housing_overhead"],
        "lens_refraction_vs_wavelength": "lens_refraction_vs_wavelength",
        "bin_reflection_vs_wavelength": "mirror_reflectivity_vs_wavelength",
        "children": []
    }

    scn = {
        "functions": [
            {
                "name": "mirror_reflectivity_vs_wavelength",
                "argument_versus_value": [
                    [2.238e-07, 1.0],
                    [7.010e-07, 1.0]
                ],
            "comment": "Ideal mirror, perfect reflection."},
            {
                "name": "lens_refraction_vs_wavelength",
                "argument_versus_value": [
                    [240e-9, 1.5133],
                    [280e-9, 1.4942],
                    [320e-9, 1.4827],
                    [360e-9, 1.4753],
                    [400e-9, 1.4701],
                    [486e-9, 1.4631],
                    [546e-9, 1.4601],
                    [633e-9, 1.4570],
                    [694e-9, 1.4554],
                    [753e-9, 1.4542]],
                "comment": "Hereaus Quarzglas GmbH and Co. KG"}
        ],
        "colors": [
            {"name":"orange", "rgb":[255, 91, 49]},
            {"name":"wood_brown", "rgb":[200, 200, 0]}
        ],
        "children": [
            {
                "type": "Frame",
                "name": "Portal",
                "pos": [0, 0, 0],
                "rot": [0, 0, 0],
                "children": []
            }
        ]
    }

    scn["children"][0]["children"].append(mirror)
    scn["children"][0]["children"].append(sensor_part)
    return scn
