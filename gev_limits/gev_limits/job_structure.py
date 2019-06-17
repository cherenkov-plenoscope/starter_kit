import numpy as np


INSTRUMENT_KEYS = [
    'aperture_radius',
    'num_paxel_on_diagonal',
    'field_of_view_radius_deg',
    'num_pixel_on_diagonal',
    'time_radius',
    'num_time_slices',
    'mirror_reflectivity',
    'photo_detection_efficiency',
]

PARTICLE_KEYS = [
    'prmpar',
    'E_start',
    'E_stop',
    'max_theta_deg',
]

SITE_KEYS = [
    'atmosphere',
    'observation_level_altitude_asl',
    'earth_magnetic_field_x_muT',
    'earth_magnetic_field_z_muT',
]


example_job = {
    'random_seed': 1,
    'trigger_threshold': 50,

    'instrument': {
        'aperture_radius': 35.5,
        'num_paxel_on_diagonal': 8,
        'field_of_view_radius_deg': 3.25,
        'num_pixel_on_diagonal': int(np.round(6.5/0.0667)),
        'time_radius': 25e-9,
        'num_time_slices': 100,
        'mirror_reflectivity': 0.8,
        'photo_detection_efficiency': 0.25,
    },

    'particle': {
        'prmpar': 1,
        'E_start': 0.8,
        'E_stop': 1.6,
        'max_theta_deg': 2.,
    },

    'site': {
        'atmosphere': 10,
        'observation_level_altitude_asl': 2347.0,
        'earth_magnetic_field_x_muT': 12.5,
        'earth_magnetic_field_z_muT': -25.9,
    },
}