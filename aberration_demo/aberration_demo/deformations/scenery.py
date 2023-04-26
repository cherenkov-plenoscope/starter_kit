import numpy as np
from .. import portal
from . import parabola_segmented


def make_plenoscope_scenery_aligned_deformed(
    mirror_dimensions,
    mirror_deformation_map,
    sensor_dimensions,
    sensor_transformation,
    num_paxel_on_pixel_diagonal,
):
    assert sensor_transformation["rot"]["repr"] == "tait_bryan"
    assert "xyz_deg" in sensor_transformation["rot"]
    sensor_rot_deg = np.array(sensor_transformation["rot"]["xyz_deg"])
    sensor_rot_rad = np.deg2rad(sensor_rot_deg)

    FACET_COLOR = "facet_color"

    sensor_frame = {
        "type": "LightFieldSensor",
        "name": "light_field_sensor",
        "pos": sensor_transformation["pos"],
        "rot": sensor_rot_rad,
        "num_paxel_on_pixel_diagonal": num_paxel_on_pixel_diagonal,
        "lens_refraction_vs_wavelength": "lens_refraction_vs_wavelength",
        "bin_reflection_vs_wavelength": "mirror_reflectivity_vs_wavelength",
        "children": [],
    }
    for key in portal.SENSOR:
        sensor_frame[key] = sensor_dimensions[key]

    mirror_frame = {
        "type": "Frame",
        "name": "Mirror",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0],
        "children": parabola_segmented.make_facets(
            mirror_dimensions=mirror_dimensions,
            mirror_deformation_map=mirror_deformation_map,
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
