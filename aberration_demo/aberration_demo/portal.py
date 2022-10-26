import numpy as np

HEXAGON_INNER_OVER_OUTER_RADIUS = np.sqrt(3) * 0.5

MIRROR = {}
MIRROR["focal_length"] = 106.5
MIRROR["outer_radius"] = 41.0
MIRROR["inner_radius"] = (
    HEXAGON_INNER_OVER_OUTER_RADIUS * MIRROR["outer_radius"]
)
MIRROR["facet_inner_hex_radius"] = 0.75
MIRROR["gap_between_facets"] = 0.025

SENSOR = {}
SENSOR["fov_radius_deg"] = 3.25
SENSOR["housing_overhead"] = 1.1
SENSOR["hex_pixel_fov_flat2flat_deg"] = 0.06667
SENSOR["num_paxel_on_diagonal"] = [1, 3, 9]

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
