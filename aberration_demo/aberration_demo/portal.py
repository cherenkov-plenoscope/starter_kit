import numpy as np


MIRROR = {
    "focal_length": 106.5,
    "max_outer_aperture_radius": 41.0,
    "min_inner_aperture_radius": 3.05,
    "outer_aperture_shape_hex": True,
    "facet_inner_hex_radius": 0.75,
    "gap_between_facets": 0.025,
}

SENSOR = {
    "expected_imaging_system_focal_length": 106.5,
    "expected_imaging_system_aperture_radius": 35.5,
    "max_FoV_diameter_deg": 6.5,
    "hex_pixel_FoV_flat2flat_deg": 0.06667,
    "housing_overhead": 1.1,
}
SENSOR_NUM_PAXEL_ON_PIXEL_DIAGONAL = 9

LENS_REFRACTION_VS_WAVELENGTH = {
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
        [753e-9, 1.4542],
    ],
    "comment": "Hereaus Quarzglas GmbH and Co. KG",
}

MIRROR_REFLECTIVITY_VS_WAVELENGTH = {
    "name": "mirror_reflectivity_vs_wavelength",
    "argument_versus_value": [[2.238e-07, 0.8], [7.010e-07, 0.8]],
    "comment": "Ideal mirror, perfect reflection.",
}
