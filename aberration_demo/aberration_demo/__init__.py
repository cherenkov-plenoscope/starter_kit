"""
simulate different light-field-sensors
"""
import numpy as np
import plenoirf
import queue_map_reduce
import multiprocessing
import json_numpy
import os

EXECUTABLES = {
    "merlict_plenoscope_propagator_path": os.path.join(
        "build", "merlict", "merlict-plenoscope-propagation"
    ),
    "merlict_plenoscope_calibration_map_path": os.path.join(
        "build", "merlict", "merlict-plenoscope-calibration-map"
    ),
    "merlict_plenoscope_calibration_reduce_path": os.path.join(
        "build", "merlict", "merlict-plenoscope-calibration-reduce"
    ),
}

CFG = {}
CFG["mirror"] = {}
CFG["mirror"]["focal_length"] = 106.5
CFG["mirror"]["inner_radius"] = 35.5
CFG["mirror"]["outer_radius"] = (2/np.sqrt(3)) * CFG["mirror"]["inner_radius"]
CFG["sensor"] = {}
CFG["sensor"]["fov_radius_deg"] = 9.0
CFG["sensor"]["housing_overhead"] = 1.1
CFG["sensor"]["hex_pixel_fov_flat2flat_deg"] = 0.1

OFF_AXIS_DEG = np.linspace(0.0, CFG["sensor"]["fov_radius_deg"] - 1.0, 3)


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
    "davies_cotton": make_mirror_davies_cotton,
    "parabola_segmented": make_mirror_parabola_segmented,
    "sphere_monolith": make_mirror_spherical_monolith,
}




SENSORS = [1, 3, 9]


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


LFG_CFG = {}
LFG_CFG["light_field_geometry"] = {}
LFG_CFG["light_field_geometry"]["num_blocks"] = 24
LFG_CFG["light_field_geometry"]["num_photons_per_block"] = 1000000

pool = multiprocessing.Pool(6)

work_dir = "demoabrr"
os.makedirs(work_dir, exist_ok=True)

for mkey in MIRRORS:
    for num_paxel_on_diagonal in SENSORS:

        scenario_key = "{:s}_paxel{:d}".format(mkey, num_paxel_on_diagonal)
        scenario_dir = os.path.join(work_dir, scenario_key)

        os.makedirs(scenario_dir, exist_ok=True)
        scenery_dir = os.path.join(scenario_dir, "input", "scenery")
        os.makedirs(scenery_dir, exist_ok=True)
        with open(os.path.join(scenery_dir, "scenery.json"), "wt") as f:
            s = make_plenoscope_scenery_for_merlict(
                mirror_key=mkey,
                num_paxel_on_diagonal=num_paxel_on_diagonal,
                cfg=CFG,
            )
            f.write(json_numpy.dumps(s))

        lfg_path = os.path.join(scenario_dir, "light_field_geometry")
        if not os.path.exists(lfg_path):
            plenoirf._estimate_light_field_geometry_of_plenoscope(
                cfg={"light_field_geometry": {
                        "num_blocks": 24 * num_paxel_on_diagonal ** 2,
                        "num_photons_per_block": 1000000,
                    }
                },
                out_absdir=scenario_dir,
                map_and_reduce_pool=pool,
                executables=EXECUTABLES,
            )
