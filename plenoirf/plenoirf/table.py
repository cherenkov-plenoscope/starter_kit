import numpy as np

STRUCTURE = {}

STRUCTURE["primary"] = {
    "particle_id": {"dtype": "<i8", "comment": "CORSIKA particle-id"},
    "energy_GeV": {"dtype": "<f8", "comment": ""},
    "azimuth_rad": {
        "dtype": "<f8",
        "comment": "Direction of the primary particle w.r.t. magnetic north.",
    },
    "zenith_rad": {
        "dtype": "<f8",
        "comment": "Direction of the primary particle.",
    },
    "max_scatter_rad": {"dtype": "<f8", "comment": ""},
    "magnet_azimuth_rad": {
        "dtype": "<f8",
        "comment": "The azimuth direction that the primary particle needs to have "
        "in order to induce an air-shower that emits its Cherenkov-light "
        "head on the pointing of the plenoscope.",
    },
    "magnet_zenith_rad": {
        "dtype": "<f8",
        "comment": "The zenith direction that the primary particle needs to have "
        "in order to induce an air-shower that emits its Cherenkov-light "
        "head on the pointing of the plenoscope.",
    },
    "magnet_cherenkov_pool_x_m": {
        "dtype": "<f8",
        "comment": "This offset must be added to the core-position, where "
        "the trajectory of the primary particle intersects the "
        "observation-level, in order for the plenoscope to stand in "
        "the typical center of the Cherenkov-pool.",
    },
    "magnet_cherenkov_pool_y_m": {"dtype": "<f8", "comment": ""},
    "solid_angle_thrown_sr": {"dtype": "<f8", "comment": ""},
    "depth_g_per_cm2": {"dtype": "<f8", "comment": ""},
    "momentum_x_GeV_per_c": {"dtype": "<f8", "comment": ""},
    "momentum_y_GeV_per_c": {"dtype": "<f8", "comment": ""},
    "momentum_z_GeV_per_c": {"dtype": "<f8", "comment": ""},
    "first_interaction_height_asl_m": {"dtype": "<f8", "comment": ""},
    "starting_height_asl_m": {"dtype": "<f8", "comment": ""},
    "starting_x_m": {"dtype": "<f8", "comment": ""},
    "starting_y_m": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["cherenkovsize"] = {
    "num_bunches": {"dtype": "<i8", "comment": ""},
    "num_photons": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["grid"] = {
    "num_bins_thrown": {
        "dtype": "<i8",
        "comment": "The number of all grid-bins which can collect Cherenkov-photons.",
    },
    "bin_width_m": {"dtype": "<f8", "comment": ""},
    "field_of_view_radius_deg": {"dtype": "<f8", "comment": ""},
    "pointing_direction_x": {"dtype": "<f8", "comment": ""},
    "pointing_direction_y": {"dtype": "<f8", "comment": ""},
    "pointing_direction_z": {"dtype": "<f8", "comment": ""},
    "random_shift_x_m": {"dtype": "<f8", "comment": ""},
    "random_shift_y_m": {"dtype": "<f8", "comment": ""},
    "magnet_shift_x_m": {"dtype": "<f8", "comment": ""},
    "magnet_shift_y_m": {"dtype": "<f8", "comment": ""},
    "total_shift_x_m": {
        "dtype": "<f8",
        "comment": "Sum of random and magnetic shift.",
    },
    "total_shift_y_m": {
        "dtype": "<f8",
        "comment": "Sum of random and magnetic shift.",
    },
    "num_bins_above_threshold": {"dtype": "<i8", "comment": ""},
    "overflow_x": {"dtype": "<i8", "comment": ""},
    "overflow_y": {"dtype": "<i8", "comment": ""},
    "underflow_x": {"dtype": "<i8", "comment": ""},
    "underflow_y": {"dtype": "<i8", "comment": ""},
    "area_thrown_m2": {"dtype": "<f8", "comment": ""},
    "artificial_core_limitation": {"dtype": "<i8", "comment": "Flag"},
    "artificial_core_limitation_radius_m": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["cherenkovpool"] = {
    "maximum_asl_m": {"dtype": "<f8", "comment": ""},
    "wavelength_median_nm": {"dtype": "<f8", "comment": ""},
    "cx_median_rad": {"dtype": "<f8", "comment": ""},
    "cy_median_rad": {"dtype": "<f8", "comment": ""},
    "x_median_m": {"dtype": "<f8", "comment": ""},
    "y_median_m": {"dtype": "<f8", "comment": ""},
    "bunch_size_median": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["cherenkovsizepart"] = STRUCTURE["cherenkovsize"].copy()
STRUCTURE["cherenkovpoolpart"] = STRUCTURE["cherenkovpool"].copy()

STRUCTURE["core"] = {
    "bin_idx_x": {"dtype": "<i8", "comment": ""},
    "bin_idx_y": {"dtype": "<i8", "comment": ""},
    "core_x_m": {"dtype": "<f8", "comment": ""},
    "core_y_m": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["trigger"] = {
    "num_cherenkov_pe": {"dtype": "<i8", "comment": ""},
    "response_pe": {"dtype": "<i8", "comment": ""},
    "focus_00_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_01_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_02_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_03_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_04_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_05_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_06_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_07_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_08_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_09_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_10_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_11_response_pe": {"dtype": "<i8", "comment": ""},
}

STRUCTURE["pasttrigger"] = {}

STRUCTURE["cherenkovclassification"] = {
    "num_true_positives": {"dtype": "<i8", "comment": ""},
    "num_false_negatives": {"dtype": "<i8", "comment": ""},
    "num_false_positives": {"dtype": "<i8", "comment": ""},
    "num_true_negatives": {"dtype": "<i8", "comment": ""},
}

STRUCTURE["features"] = {}
STRUCTURE["features"]["num_photons"] = {
    "dtype": "<i8",
    "comment": "The number of photon-eqivalents that are identified to be dense cluster(s) of Cherenkov-photons",
    "histogram": "geomspace",
    "unit": "1",
}
STRUCTURE["features"]["paxel_intensity_peakness_std_over_mean"] = {
    "dtype": "<f8",
    "comment": "A measure for the intensity distribution on the aperture-plane. The larger the value, the less evenly the intensity is distributed on the plane.",
    "histogram": "geomspace",
    "unit": "1",
}
STRUCTURE["features"]["paxel_intensity_peakness_max_over_mean"] = {
    "dtype": "<f8",
    "comment": "A measure for the intensity distribution on the aperture-plane. The larger the value, the more the intensity is concentrated in a small area on the aperture-plane.",
    "histogram": "geomspace",
    "unit": "1",
}

paxel_intensity_median_str = "Median intersection-positions in {:s} of reconstructed Cherenkov-photons on the aperture-plane"
STRUCTURE["features"]["paxel_intensity_median_x"] = {
    "dtype": "<f8",
    "comment": paxel_intensity_median_str.format("x"),
    "histogram": "linspace",
    "unit": "m",
}
STRUCTURE["features"]["paxel_intensity_median_y"] = {
    "dtype": "<f8",
    "comment": paxel_intensity_median_str.format("y"),
    "histogram": "linspace",
    "unit": "m",
}

_watershed_str = "A measure for the areal distribution of reconstructed Cherenkov-photons on the aperture-plane."
STRUCTURE["features"]["aperture_num_islands_watershed_rel_thr_2"] = {
    "dtype": "<i8",
    "comment": _watershed_str,
    "histogram": "linspace",
    "unit": "1",
}
STRUCTURE["features"]["aperture_num_islands_watershed_rel_thr_4"] = {
    "dtype": "<i8",
    "comment": _watershed_str,
    "histogram": "linspace",
    "unit": "1",
}
STRUCTURE["features"]["aperture_num_islands_watershed_rel_thr_8"] = {
    "dtype": "<i8",
    "comment": _watershed_str,
    "histogram": "linspace",
    "unit": "1",
}

_light_front_c_str = "Incident-direction in {:s} of reconstructed Cherenkov-photon-plane passing through the aperture-plane."
STRUCTURE["features"]["light_front_cx"] = {
    "dtype": "<f8",
    "comment": _light_front_c_str.format("x"),
    "histogram": "linspace",
    "unit": "rad",
}
STRUCTURE["features"]["light_front_cy"] = {
    "dtype": "<f8",
    "comment": _light_front_c_str.format("y"),
    "histogram": "linspace",
    "unit": "rad",
}

_image_infinity_c_mean_str = "Mean incident-direction in {:s} of reconstructed Cherenkov-photons in the image focussed to infinity."
STRUCTURE["features"]["image_infinity_cx_mean"] = {
    "dtype": "<f8",
    "comment": _image_infinity_c_mean_str.format("x"),
    "histogram": "linspace",
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_cy_mean"] = {
    "dtype": "<f8",
    "comment": _image_infinity_c_mean_str.format("y"),
    "histogram": "linspace",
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_cx_std"] = {
    "dtype": "<f8",
    "comment": "",
    "histogram": "geomspace",
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_cy_std"] = {
    "dtype": "<f8",
    "comment": "",
    "histogram": "geomspace",
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_num_photons_on_edge_field_of_view"] = {
    "dtype": "<i8",
    "comment": "Number of photon-eqivalents on the edge of the field-of-view in an image focused on infinity.",
    "histogram": "linspace",
    "unit": "p.e.",
}
STRUCTURE["features"]["image_smallest_ellipse_object_distance"] = {
    "dtype": "<f8",
    "comment": "The object-distance in front of the aperture where the refocused image of the airshower yields the Hillas-ellipse with the smallest solid angle. See also 'image_smallest_ellipse_solid_angle'.",
    "histogram": "geomspace",
    "unit": "m",
}
STRUCTURE["features"]["image_smallest_ellipse_solid_angle"] = {
    "dtype": "<f8",
    "comment": "The solid angle of the smallest Hillas-ellipse in all refocused images. See also 'image_smallest_ellipse_object_distance'.",
    "histogram": "geomspace",
    "unit": "sr",
}
STRUCTURE["features"]["image_smallest_ellipse_half_depth"] = {
    "dtype": "<f8",
    "comment": "The range in object-distance for the Hillas-ellipse to double its solid angle when refocusing starts at the smallest ellipse.",
    "histogram": "geomspace",
    "unit": "m",
}

image_half_depth_shift_c_str = "How much the mean intensity in the image shifts in {:s} when refocussing from smallest to double solid angle of ellipse."
STRUCTURE["features"]["image_half_depth_shift_cx"] = {
    "dtype": "<f8",
    "comment": image_half_depth_shift_c_str.format("cx"),
    "histogram": "linspace",
    "unit": "rad",
}
STRUCTURE["features"]["image_half_depth_shift_cy"] = {
    "dtype": "<f8",
    "comment": image_half_depth_shift_c_str.format("cy"),
    "histogram": "linspace",
    "unit": "rad",
}
STRUCTURE["features"][
    "image_smallest_ellipse_num_photons_on_edge_field_of_view"
] = {
    "dtype": "<i8",
    "comment": "Number of photon-eqivalents on the edge of the field-of-view in an image focused to the smallest Hillas-ellipse.",
    "histogram": "geomspace",
    "unit": "p.e.",
}
STRUCTURE["features"]["image_num_islands"] = {
    "dtype": "<i8",
    "comment": "The number of individual dense clusters of reconstructed Cherenkov-photons in the image-space.",
    "histogram": "linspace",
    "unit": "1",
}
