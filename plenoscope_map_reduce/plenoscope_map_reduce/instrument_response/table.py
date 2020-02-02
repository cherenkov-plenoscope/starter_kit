import pandas as pd
import numpy as np
import shutil
import tarfile
import io
import glob
import os
from os import path as op

MAX_NUM_EVENTS_IN_RUN = 1000


def random_seed_based_on(run_id, airshower_id):
    return run_id*MAX_NUM_EVENTS_IN_RUN + airshower_id


CONFIG = {
    'index': {
        'run_id': {'dtype': '<i8', 'comment': ''},
        'airshower_id': {'dtype': '<i8', 'comment': ''},
    },
    'levels': {}
}

CONFIG['levels']['primary'] = {
    'particle_id': {'dtype': '<i8', 'comment': 'CORSIKA particle-id'},
    'energy_GeV': {'dtype': '<f8', 'comment': ''},
    'azimuth_rad': {'dtype': '<f8', 'comment': 'w.r.t. magnetic north.'},
    'zenith_rad': {'dtype': '<f8', 'comment': ''},
    'max_scatter_rad': {'dtype': '<f8', 'comment': ''},
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

CONFIG['levels']['cherenkovsize'] = {
    'num_bunches': {'dtype': '<i8', 'comment': ''},
    'num_photons': {'dtype': '<f8', 'comment': ''},
}

CONFIG['levels']['grid'] = {
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

CONFIG['levels']['cherenkovpool'] = {
    'maximum_asl_m': {'dtype': '<f8', 'comment': ''},
    'wavelength_median_nm': {'dtype': '<f8', 'comment': ''},
    'cx_median_rad': {'dtype': '<f8', 'comment': ''},
    'cy_median_rad': {'dtype': '<f8', 'comment': ''},
    'x_median_m': {'dtype': '<f8', 'comment': ''},
    'y_median_m': {'dtype': '<f8', 'comment': ''},
    'bunch_size_median': {'dtype': '<f8', 'comment': ''},
}

CONFIG['levels']['cherenkovsizepart'] = CONFIG[
    'levels']['cherenkovsize'].copy()
CONFIG['levels']['cherenkovpoolpart'] = CONFIG[
    'levels']['cherenkovpool'].copy()

CONFIG['levels']['core'] = {
    'bin_idx_x': {'dtype': '<i8', 'comment': ''},
    'bin_idx_y': {'dtype': '<i8', 'comment': ''},
    'core_x_m': {'dtype': '<f8', 'comment': ''},
    'core_y_m': {'dtype': '<f8', 'comment': ''},
}

CONFIG['levels']['trigger'] = {
    'num_cherenkov_pe': {'dtype': '<i8', 'comment': ''},
    'response_pe': {'dtype': '<i8', 'comment': ''},
    'refocus_0_object_distance_m': {'dtype': '<f8', 'comment': ''},
    'refocus_0_respnse_pe': {'dtype': '<i8', 'comment': ''},
    'refocus_1_object_distance_m': {'dtype': '<f8', 'comment': ''},
    'refocus_1_respnse_pe': {'dtype': '<i8', 'comment': ''},
    'refocus_2_object_distance_m': {'dtype': '<f8', 'comment': ''},
    'refocus_2_respnse_pe': {'dtype': '<i8', 'comment': ''},
}

CONFIG['levels']['pasttrigger'] = {
}

CONFIG['levels']['cherenkovclassification'] = {
    'num_true_positives': {'dtype': '<i8', 'comment': ''},
    'num_false_negatives': {'dtype': '<i8', 'comment': ''},
    'num_false_positives': {'dtype': '<i8', 'comment': ''},
    'num_true_negatives': {'dtype': '<i8', 'comment': ''},
}

CONFIG['levels']['features'] = {
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

FORMAT_SUFFIX = 'csv'
CONFIG_LEVELS_KEYS = list(CONFIG['levels'].keys())


def merge_level(
    event_table,
    level_a,
    level_b,
    keys_a=None,
    keys_b=None,
    on=["run_id", "airshower_id"]
):
    a_df = pd.DataFrame(event_table[level_a])
    if keys_a is not None:
        a_df = a_df[on + keys_a]
    b_df = pd.DataFrame(event_table[level_b])
    if keys_b is not None:
        b_df = b_df[on + keys_b]

    a_rename = {}
    for key in list(a_df.columns):
        if key not in on:
            a_rename[key] = "{:s}.{:s}".format(level_a, key)
    a_df = a_df.rename(columns=a_rename)
    b_rename = {}
    for key in list(b_df.columns):
        if key not in on:
            b_rename[key] = "{:s}.{:s}".format(level_b, key)
    b_df = b_df.rename(columns=b_rename)

    return pd.merge(
        pd.DataFrame(a_df),
        pd.DataFrame(b_df),
        on=on).to_records(index=False)


def _empty_recarray(config, level):
    dtypes = []
    for k in config['index']:
        dtypes.append((k, config['index'][k]['dtype']))
    for k in config['levels'][level]:
        dtypes.append((k, config['levels'][level][k]['dtype']))
    return np.rec.array(
        obj=np.array([]),
        dtype=dtypes)


def _assert_same_keys(keys_a, keys_b):
    uni_keys = list(set(keys_a + keys_b))
    for key in uni_keys:
        assert key in keys_a and key in keys_b, 'Key: {:s}'.format(key)


def _expected_keys(config, level):
    return (
        list(config['index'].keys()) +
        list(config['levels'][level].keys()))


def _assert_recarray_keys(rec, config, level):
    rec_keys = list(rec.dtype.names)
    expected_keys = _expected_keys(config=config, level=level)
    _assert_same_keys(rec_keys, expected_keys)
    for index_key in config['index']:
        rec_dtype = rec.dtype[index_key]
        exp_dtype = np.dtype(config['index'][index_key]['dtype'])
        assert rec_dtype == exp_dtype, (
            'Wrong dtype for index-key: "{:s}", on level {:s}'.format(
                rec_key, level))
    for rec_key in config['levels'][level]:
        rec_dtype = rec.dtype[rec_key]
        exp_dtype = np.dtype(config['levels'][level][rec_key]['dtype'])
        assert rec_dtype == exp_dtype, (
            'Wrong dtype for key: "{:s}", on level {:s}'.format(
                rec_key, level))


def level_records_to_csv(level_records, config, level):
    expected_keys = _expected_keys(config=config, level=level)
    # assert keys are valid
    if len(level_records) > 0:
        for one_dict in level_records:
            one_dict_keys = list(one_dict.keys())
            _assert_same_keys(one_dict_keys, expected_keys)
        df = pd.DataFrame(level_records)
    else:
        df = pd.DataFrame(columns=expected_keys)
    return df.to_csv(index=False)


def df_to_recarray(df, config, level):
    expected_keys = _expected_keys(config=config, level=level)
    if len(df) > 0:
        rec = df.to_records(index=False)
    else:
        rec = _empty_recarray(config=config, level=level)
    _assert_recarray_keys(rec=rec, config=config, level=level)
    return rec


def run_ids_in_dir(feature_dir, wild_card):
    paths = glob.glob(op.join(feature_dir, wild_card))
    run_ids = []
    for path in paths:
        basename = op.basename(path)
        run_ids.append(int(basename[0:6]))
    return list(set(run_ids))


def reduce_feature_dir(
    feature_dir,
    format_suffix=FORMAT_SUFFIX,
    config=CONFIG,
):
    evttab = {}
    for level in config['levels']:
        evttab[level] = []

    for run_id in run_ids_in_dir(feature_dir, wild_card= '*_event_table.tar'):
        rpath = op.join(feature_dir, '{:06d}_event_table.tar'.format(run_id))
        num_levels = 0
        with tarfile.open(rpath, "r") as tarfin:
            for tarinfo in tarfin:
                level, format_suffix = str.split(tarinfo.name, '.')
                assert level in config['levels']
                assert format_suffix == FORMAT_SUFFIX
                level_df = pd.read_csv(tarfin.extractfile(tarinfo))
                level_recarray = df_to_recarray(
                    df=level_df,
                    config=config,
                    level=level)
                evttab[level].append(level_recarray)
                num_levels += 1
        assert num_levels == len(config["levels"])
    for level in config['levels']:
        ll = evttab[level]
        cat = np.concatenate(ll)
        evttab[level] = cat
    return evttab


def write_site_particle(
    path,
    event_table,
    config=CONFIG,
    format_suffix=FORMAT_SUFFIX
):
    assert op.splitext(path)[1] == '.tar'
    with tarfile.open(path, 'w') as tarout:
        for level in config['levels']:
            level_df = pd.DataFrame(event_table[level])
            level_csv = level_df.to_csv(index=False)
            level_filename = '{:s}.{:s}'.format(level, format_suffix)
            with io.BytesIO() as fbuff:
                fbuff.write(str.encode(level_csv))
                fbuff.seek(0)
                tarinfo = tarfile.TarInfo(name=level_filename)
                tarinfo.size = len(fbuff.getvalue())
                tarout.addfile(tarinfo=tarinfo, fileobj=fbuff)


def read_site_particle(
    path,
    config=CONFIG,
    format_suffix=FORMAT_SUFFIX
):
    evttab = {}
    num_levels = 0
    with tarfile.open(path, "r") as tarfin:
        for tarinfo in tarfin:
            level, format_suffix = str.split(tarinfo.name, '.')
            assert level in config['levels']
            assert format_suffix == FORMAT_SUFFIX
            level_df = pd.read_csv(tarfin.extractfile(tarinfo))
            evttab[level] = df_to_recarray(
                df=level_df,
                config=config,
                level=level)
            num_levels += 1
    assert num_levels == len(config["levels"])
    return evttab


def write_level(path, level_records, config, level):
    csv = level_records_to_csv(level_records, config, level)
    with open(path+'.tmp', 'wt') as f:
        f.write(csv)
    shutil.move(path+'.tmp', path)
