import pandas as pd
import numpy as np
import shutil
import tarfile
import io
import glob
import os


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


def write_level(path, list_of_dicts, config, level):
    expected_keys = _expected_keys(config=config, level=level)
    # assert keys are valid
    if len(list_of_dicts) > 0:
        for one_dict in list_of_dicts:
            one_dict_keys = list(one_dict.keys())
            _assert_same_keys(one_dict_keys, expected_keys)
        df = pd.DataFrame(list_of_dicts)
    else:
        df = pd.DataFrame(columns=expected_keys)
    with open(path+'.tmp', 'wt') as f:
        f.write(df.to_csv(index=False))
    shutil.move(path+'.tmp', path)


def read_level(path, config, level):
    expected_keys = _expected_keys(config=config, level=level)
    df = pd.read_csv(path)
    if len(df) > 0:
        rec = df.to_records(index=False)
    else:
        rec = _empty_recarray(config=config, level=level)
    _assert_recarray_keys(rec=rec, config=config, level=level)
    return rec


def _run_ids_in_dir(feature_dir, wild_card):
    paths = glob.glob(os.path.join(feature_dir, wild_card))
    run_ids = []
    for path in paths:
        basename = os.path.basename(path)
        run_ids.append(int(basename[0:6]))
    return list(set(run_ids))


def reduce_site_particle(
    site_particle_feature_dir,
    format_suffix=FORMAT_SUFFIX,
    config=CONFIG,
):
    wild_card = '*.{:s}'.format(format_suffix)
    spf_dir = site_particle_feature_dir
    table = {}
    for level in config['levels']:
        table[level] = []
        for run_id in _run_ids_in_dir(spf_dir, wild_card=wild_card):
            fpath = op.join(
                feature_dir,
                '{:06d}_{:s}.{:s}'.format(run_id, level, format_suffix))
            _rec = read_level(
                path=fpath,
                config=config,
                level=level)
            table[level].append(_rec)
    for level in config['levels']:
        print(particle_key, level)
        ll = table[level]
        cat = np.concatenate(ll)
        table[level] = cat
    return table


def write_site_particle(
    path,
    table,
    config=CONFIG,
    format_suffix=FORMAT_SUFFIX
):
    assert op.splitext(path)[1] == '.tar'
    with tarfile.open(path, 'w') as tarout:
        for level in config['levels']:
            level_df = pd.DataFrame(table[level])
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
    table = {}
    with tarfile.open(path, 'r') as tarin:
        for level in config['levels']:
            level_filename = '{:s}.{:s}'.format(level, format_suffix)
            tarinfo = tarin.getmember(level_filename)
            with io.BytesIO() as fbuff:
                fbuff.write(tarin.extractfile(tarinfo).read())
                fbuff.seek(0)
                table[level] = read_level(
                    fbuff,
                    config=config,
                    level=level)
    return table
