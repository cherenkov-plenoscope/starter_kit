import pandas as pd
import numpy as np
import shutil
import tarfile
import io
import glob
import os
import msgpack
from os import path as op

NUM_DIGITS_RUN_ID = 6
NUM_DIGITS_AIRSHOWER_ID = 3
NUM_DIGITS_SEED = NUM_DIGITS_RUN_ID + NUM_DIGITS_AIRSHOWER_ID

NUM_AIRSHOWER_IDS_IN_RUN = 10**NUM_DIGITS_AIRSHOWER_ID
NUM_RUN_IDS = 10**NUM_DIGITS_RUN_ID
NUM_SEEDS = NUM_AIRSHOWER_IDS_IN_RUN*NUM_RUN_IDS

SEED_TEMPLATE_STR = '{seed:0'+str(NUM_DIGITS_SEED)+'d}'


def random_seed_based_on(run_id, airshower_id):
    assert is_valid_run_id(run_id)
    assert is_valid_airshower_id(airshower_id)
    return run_id*NUM_AIRSHOWER_IDS_IN_RUN + airshower_id


def run_id_from_seed(seed):
    if np.isscalar(seed):
        assert seed <= NUM_SEEDS
    else:
        seed = np.array(seed)
        assert (seed <= NUM_SEEDS).all()
    return seed//NUM_AIRSHOWER_IDS_IN_RUN


def airshower_id_from_seed(seed):
    return seed - run_id_from_seed(seed)*NUM_AIRSHOWER_IDS_IN_RUN


def is_valid_run_id(run_id):
    if run_id >= 0 and run_id < NUM_RUN_IDS:
        return True
    else:
        return False


def is_valid_airshower_id(airshower_id):
    if airshower_id >= 0 and airshower_id < NUM_AIRSHOWER_IDS_IN_RUN:
        return True
    else:
        return False


STRUCTURE = {}

STRUCTURE['primary'] = {
    'particle_id': {'dtype': '<i8', 'comment': 'CORSIKA particle-id'},
    'energy_GeV': {'dtype': '<f8', 'comment': ''},
    'azimuth_rad': {'dtype': '<f8', 'comment':
        'Direction of the primary particle w.r.t. magnetic north.'},
    'zenith_rad': {'dtype': '<f8', 'comment':
        'Direction of the primary particle.'},
    'max_scatter_rad': {'dtype': '<f8', 'comment': ''},

    'magnet_azimuth_rad': {'dtype': '<f8', 'comment':
        'The azimuth direction that the primary particle needs to have '
        'in order to induce an air-shower that emits its Cherenkov-light '
        'head on the pointing of the plenoscope.'},
    'magnet_zenith_rad': {'dtype': '<f8', 'comment':
        'The zenith direction that the primary particle needs to have '
        'in order to induce an air-shower that emits its Cherenkov-light '
        'head on the pointing of the plenoscope.'},
    'magnet_cherenkov_pool_x_m': {'dtype': '<f8', 'comment':
        'This offset must be added to the core-position, where '
        'the trajectory of the primary particle intersects the '
        'observation-level, in order for the plenoscope to stand in '
        'the typical center of the Cherenkov-pool.'},
    'magnet_cherenkov_pool_y_m': {'dtype': '<f8', 'comment': ''},

    'solid_angle_thrown_sr': {'dtype': '<f8', 'comment': ''},
    'depth_g_per_cm2': {'dtype': '<f8', 'comment': ''},
    'momentum_x_GeV_per_c': {'dtype': '<f8', 'comment': ''},
    'momentum_y_GeV_per_c': {'dtype': '<f8', 'comment': ''},
    'momentum_z_GeV_per_c': {'dtype': '<f8', 'comment': ''},
    'first_interaction_height_asl_m': {'dtype': '<f8', 'comment': ''},
    'starting_height_asl_m': {'dtype': '<f8', 'comment': ''},
    'starting_x_m': {'dtype': '<f8', 'comment': ''},
    'starting_y_m': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['cherenkovsize'] = {
    'num_bunches': {'dtype': '<i8', 'comment': ''},
    'num_photons': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['grid'] = {
    'num_bins_radius': {'dtype': '<i8', 'comment': ''},
    'plenoscope_diameter_m': {'dtype': '<f8', 'comment': ''},
    'plenoscope_field_of_view_radius_deg': {'dtype': '<f8', 'comment': ''},
    'plenoscope_pointing_direction_x': {'dtype': '<f8', 'comment': ''},
    'plenoscope_pointing_direction_y': {'dtype': '<f8', 'comment': ''},
    'plenoscope_pointing_direction_z': {'dtype': '<f8', 'comment': ''},
    'random_shift_x_m': {'dtype': '<f8', 'comment': ''},
    'random_shift_y_m': {'dtype': '<f8', 'comment': ''},
    'hist_00': {'dtype': '<i8', 'comment': ''},
    'hist_01': {'dtype': '<i8', 'comment': ''},
    'hist_02': {'dtype': '<i8', 'comment': ''},
    'hist_03': {'dtype': '<i8', 'comment': ''},
    'hist_04': {'dtype': '<i8', 'comment': ''},
    'hist_05': {'dtype': '<i8', 'comment': ''},
    'hist_06': {'dtype': '<i8', 'comment': ''},
    'hist_07': {'dtype': '<i8', 'comment': ''},
    'hist_08': {'dtype': '<i8', 'comment': ''},
    'hist_09': {'dtype': '<i8', 'comment': ''},
    'hist_10': {'dtype': '<i8', 'comment': ''},
    'hist_11': {'dtype': '<i8', 'comment': ''},
    'hist_12': {'dtype': '<i8', 'comment': ''},
    'hist_13': {'dtype': '<i8', 'comment': ''},
    'hist_14': {'dtype': '<i8', 'comment': ''},
    'hist_15': {'dtype': '<i8', 'comment': ''},
    'hist_16': {'dtype': '<i8', 'comment': ''},
    'num_bins_above_threshold': {'dtype': '<i8', 'comment': ''},
    'overflow_x': {'dtype': '<i8', 'comment': ''},
    'overflow_y': {'dtype': '<i8', 'comment': ''},
    'underflow_x': {'dtype': '<i8', 'comment': ''},
    'underflow_y': {'dtype': '<i8', 'comment': ''},
    'area_thrown_m2': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['cherenkovpool'] = {
    'maximum_asl_m': {'dtype': '<f8', 'comment': ''},
    'wavelength_median_nm': {'dtype': '<f8', 'comment': ''},
    'cx_median_rad': {'dtype': '<f8', 'comment': ''},
    'cy_median_rad': {'dtype': '<f8', 'comment': ''},
    'x_median_m': {'dtype': '<f8', 'comment': ''},
    'y_median_m': {'dtype': '<f8', 'comment': ''},
    'bunch_size_median': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['cherenkovsizepart'] = STRUCTURE['cherenkovsize'].copy()
STRUCTURE['cherenkovpoolpart'] = STRUCTURE['cherenkovpool'].copy()

STRUCTURE['core'] = {
    'bin_idx_x': {'dtype': '<i8', 'comment': ''},
    'bin_idx_y': {'dtype': '<i8', 'comment': ''},
    'core_x_m': {'dtype': '<f8', 'comment': ''},
    'core_y_m': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['trigger'] = {
    'num_cherenkov_pe': {'dtype': '<i8', 'comment': ''},
    'response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_00_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_01_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_02_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_03_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_04_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_05_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_06_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_07_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_08_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_09_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_10_response_pe': {'dtype': '<i8', 'comment': ''},
    'focus_11_response_pe': {'dtype': '<i8', 'comment': ''},
}

STRUCTURE['pasttrigger'] = {}

STRUCTURE['cherenkovclassification'] = {
    'num_true_positives': {'dtype': '<i8', 'comment': ''},
    'num_false_negatives': {'dtype': '<i8', 'comment': ''},
    'num_false_positives': {'dtype': '<i8', 'comment': ''},
    'num_true_negatives': {'dtype': '<i8', 'comment': ''},
}

STRUCTURE['features'] = {
    'num_photons': {'dtype': '<i8', 'comment': ''},
    'paxel_intensity_peakness_std_over_mean': {'dtype': '<f8', 'comment': ''},
    'paxel_intensity_peakness_max_over_mean': {'dtype': '<f8', 'comment': ''},
    'paxel_intensity_median_x': {'dtype': '<f8', 'comment': ''},
    'paxel_intensity_median_y': {'dtype': '<f8', 'comment': ''},
    'aperture_num_islands_watershed_rel_thr_2':
        {'dtype': '<i8', 'comment': ''},
    'aperture_num_islands_watershed_rel_thr_4':
        {'dtype': '<i8', 'comment': ''},
    'aperture_num_islands_watershed_rel_thr_8':
        {'dtype': '<i8', 'comment': ''},
    'light_front_cx': {'dtype': '<f8', 'comment': ''},
    'light_front_cy': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cx_mean': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cy_mean': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cx_std': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cy_std': {'dtype': '<f8', 'comment': ''},
    'image_infinity_num_photons_on_edge_field_of_view':
        {'dtype': '<i8', 'comment': ''},
    'image_smallest_ellipse_object_distance': {'dtype': '<f8', 'comment': ''},
    'image_smallest_ellipse_solid_angle': {'dtype': '<f8', 'comment': ''},
    'image_smallest_ellipse_half_depth': {'dtype': '<f8', 'comment': ''},
    'image_half_depth_shift_cx': {'dtype': '<f8', 'comment': ''},
    'image_half_depth_shift_cy': {'dtype': '<f8', 'comment': ''},
    'image_smallest_ellipse_num_photons_on_edge_field_of_view':
        {'dtype': '<i8', 'comment': ''},
    'image_num_islands': {'dtype': '<i8', 'comment': ''},
}
