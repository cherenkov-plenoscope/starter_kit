from .. import portal
from . import parabola_segmented


def make_plenoscope_scenery_aligned_deformed(
    mirror_config,
    deformation_polynom,
    sensor_config,
    num_paxel_on_diagonal,
):
    FACET_COLOR = 'facet_color'

    scfg = sensor_config
    mcfg = mirror_config

    sensor_frame = {
        "type": "LightFieldSensor",
        "name": "light_field_sensor",
        "pos": [0, 0, mcfg["focal_length"]],
        "rot": [0, 0, 0],
        "expected_imaging_system_focal_length": mcfg["focal_length"],
        "expected_imaging_system_aperture_radius": mcfg["inner_radius"],
        "max_FoV_diameter_deg": 2.0 * scfg["fov_radius_deg"],
        "hex_pixel_FoV_flat2flat_deg": scfg["hex_pixel_fov_flat2flat_deg"],
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
        "children": parabola_segmented.make_facets(
            mirror_config=mirror_config,
            deformation_polynom=deformation_polynom,
            reflection_vs_wavelength='mirror_reflectivity_vs_wavelength',
            color=FACET_COLOR,
        )
    }

    scn = {
        "functions": [
            portal.MIRROR_REFLECTIVITY_VS_WAVELENGTH,
            portal.LENS_REFRACTION_VS_WAVELENGTH,
        ],
        "colors": [
            {"name": FACET_COLOR, "rgb": [255, 91, 49]},
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
