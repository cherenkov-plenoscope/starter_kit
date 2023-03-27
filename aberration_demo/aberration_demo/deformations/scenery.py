from .. import portal
from . import parabola_segmented


def make_plenoscope_scenery_aligned_deformed(
    mirror_config, mirror_deformation, sensor_config, num_paxel_on_diagonal,
):
    FACET_COLOR = "facet_color"

    sensor_frame = {
        "type": "LightFieldSensor",
        "name": "light_field_sensor",
        "pos": [0, 0, sensor_config["expected_imaging_system_focal_length"]],
        "rot": [0, 0, 0],
        "lens_refraction_vs_wavelength": "lens_refraction_vs_wavelength",
        "bin_reflection_vs_wavelength": "mirror_reflectivity_vs_wavelength",
        "children": [],
    }
    for key in portal.SENSOR:
        sensor_frame[key] = sensor_config[key]

    mirror_frame = {
        "type": "Frame",
        "name": "Mirror",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0],
        "children": parabola_segmented.make_facets(
            mirror_config=mirror_config,
            mirror_deformation=mirror_deformation,
            reflection_vs_wavelength="mirror_reflectivity_vs_wavelength",
            color=FACET_COLOR,
        ),
    }

    scn = {
        "functions": [
            portal.MIRROR_REFLECTIVITY_VS_WAVELENGTH,
            portal.LENS_REFRACTION_VS_WAVELENGTH,
        ],
        "colors": [{"name": FACET_COLOR, "rgb": [255, 91, 49]},],
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
