import numpy as np


INSTRUMENT_KEYS = [
    'aperture_radius',
    'num_paxel_on_diagonal',
    'field_of_view_radius_deg',
    'num_pixel_on_diagonal',
    'time_radius',
    'num_time_slices',
    'relative_arrival_times_std',
    'mirror_reflectivity',
    'photo_detection_efficiency',
]

PARTICLE_KEYS = [
    'prmpar',
    'E_start',
    'E_stop',
    'E_slope',
    'max_zenith_angle_deg',
    'max_scatter_radius',
]

SITE_KEYS = [
    'atmosphere',
    'observation_level_altitude_asl',
    'earth_magnetic_field_x_muT',
    'earth_magnetic_field_z_muT',
]


example_particle = {
    'prmpar': 1,
    'max_zenith_angle_deg': 2.,
    "energy":             [0.23, 0.8, 3.0, 35],
    "max_scatter_radius": [150,  150, 460, 1100]
}


example_job = {
    'random_seed': 1,
    'trigger_threshold': 103,
    'nsb_rate_pixel': 7.65,

    'instrument': {
        'aperture_radius': 35.5,
        'num_paxel_on_diagonal': 8,
        'field_of_view_radius_deg': 3.25,
        'num_pixel_on_diagonal': int(np.round(6.5/0.0667)),
        'time_radius': 25e-9,
        'num_time_slices': 100,
        'relative_arrival_times_std': 1e-9,
        'mirror_reflectivity': 0.9,
        'photo_detection_efficiency': 0.30,
    },

    'particle': {
        'prmpar': 1,
        'E_start': 0.8,
        'E_stop': 20,
        'E_slope': -1.,
        'max_zenith_angle_deg': 2.,
        'max_scatter_radius': 150,
    },

    'site': {
        'atmosphere': 10,
        'observation_level_altitude_asl': 2347.0,
        'earth_magnetic_field_x_muT': 12.5,
        'earth_magnetic_field_z_muT': -25.9,
    },
}
