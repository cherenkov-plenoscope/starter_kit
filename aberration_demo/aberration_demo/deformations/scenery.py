from .. import portal


def make_plenoscope_scenery_aligned_deformed(
    mirror_facets,
    mirror_focal_length,
    mirror_inner_radius,
    sensor_config,
):
    scfg = sensor_config

    sensor_frame = {
        "type": "LightFieldSensor",
        "name": "light_field_sensor",
        "pos": [0, 0, mirror_focal_length],
        "rot": [0, 0, 0],
        "expected_imaging_system_focal_length": mirror_focal_length,
        "expected_imaging_system_aperture_radius": mirror_inner_radius,
        "max_FoV_diameter_deg": 2.0 * scfg["fov_radius_deg"],
        "hex_pixel_FoV_flat2flat_deg": scfg[
            "hex_pixel_fov_flat2flat_deg"
        ],
        "num_paxel_on_pixel_diagonal": num_paxel_on_diagonal,
        "housing_overhead": scfg["housing_overhead"],
        "lens_refraction_vs_wavelength": "lens_refraction_vs_wavelength",
        "bin_reflection_vs_wavelength": "mirror_reflectivity_vs_wavelength",
        "children": [],
    }

    mirror_frame = {
        "type": "Frame",
        "name": "Mirror",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0],
        "children": [],
    }

    scn = {
        "functions": [
            portal.MIRROR_REFLECTIVITY,
            portal.LENS_REFRACTION,
        ],
        "colors": [
            {"name": "orange", "rgb": [255, 91, 49]},
            {"name": "wood_brown", "rgb": [200, 200, 0]},
        ],
        "children": [
            {
                "type": "Frame",
                "name": "Portal",
                "pos": [0, 0, 0],
                "rot": [0, 0, 0],
                "children": [mirror_frame, sensor_frame],
            }
        ],
    }

    return scn
