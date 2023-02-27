import numpy as np


def init_records():
    tabrec = {}
    for level_key in STRUCTURE:
        tabrec[level_key] = []
    return tabrec


STRUCTURE = {}

STRUCTURE["base"] = {
    "primary_particle_id": {"dtype": "<i8", "comment": ""},
    "primary_energy_GeV": {"dtype": "<f8", "comment": ""},
    "primary_azimuth_rad": {"dtype": "<f8", "comment": "",},
    "primary_zenith_rad": {"dtype": "<f8", "comment": "",},
    #
    "primary_start_x_m": {"dtype": "<f8", "comment": ""},
    "primary_start_y_m": {"dtype": "<f8", "comment": ""},
    "primary_start_z_m": {"dtype": "<f8", "comment": ""},
    #
    "primary_direction_x": {"dtype": "<f8", "comment": ""},
    "primary_direction_y": {"dtype": "<f8", "comment": ""},
    "primary_direction_z": {"dtype": "<f8", "comment": ""},
    #
    "instrument_x_m": {"dtype": "<f8", "comment": ""},
    "instrument_y_m": {"dtype": "<f8", "comment": ""},
    "instrument_z_m": {"dtype": "<f8", "comment": ""},
    #
    "instrument_azimuth_rad": {"dtype": "<f8", "comment": ""},
    "instrument_zenith_rad": {"dtype": "<f8", "comment": ""},
    #
    "primary_distance_to_closest_point_to_instrument_m": {
        "dtype": "<f8",
        "comment": "",
    },
    "primary_time_to_closest_point_to_instrument_s": {
        "dtype": "<f8",
        "comment": "",
    },
}

STRUCTURE["cherenkov_size"] = {
    "num_bunches": {"dtype": "<i8", "comment": ""},
    "num_photons": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["cherenkov_pool"] = {
    "maximum_asl_m": {"dtype": "<f8", "comment": ""},
    "wavelength_median_nm": {"dtype": "<f8", "comment": ""},
    "cx_median_rad": {"dtype": "<f8", "comment": ""},
    "cy_median_rad": {"dtype": "<f8", "comment": ""},
    "x_median_m": {"dtype": "<f8", "comment": ""},
    "y_median_m": {"dtype": "<f8", "comment": ""},
    "bunch_size_median": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["cherenkov_visible_size"] = STRUCTURE["cherenkov_size"].copy()
STRUCTURE["cherenkov_visible_pool"] = STRUCTURE["cherenkov_pool"].copy()

STRUCTURE["cherenkov_detected_size"] = STRUCTURE["cherenkov_size"].copy()
STRUCTURE["cherenkov_detected_pool"] = STRUCTURE["cherenkov_pool"].copy()

STRUCTURE["reconstruction"] = {
    "arrival_time_median_s": {"dtype": "<f8", "comment": ""},
    "arrival_time_stddev_s": {"dtype": "<f8", "comment": ""},
}
