import os

SITE_CHILE = {
    "earth_magnetic_field_x_muT": 20.815,
    "earth_magnetic_field_z_muT": -11.366,
    "observation_level_asl_m": 5e3,
    "atmosphere_id": 26,
}

SITE_NAMIBIA = {
    'earth_magnetic_field_x_muT': 12.5,
    'earth_magnetic_field_z_muT': -25.9,
    'observation_level_asl_m': 2300,
    'atmosphere_id': 10
}

CORSIKA_PRIMARY_MOD_PATH = os.path.abspath(
    os.path.join(
        'build',
        'corsika',
        'modified',
        'corsika-75600',
        'run',
        'corsika75600Linux_QGSII_urqmd'))
