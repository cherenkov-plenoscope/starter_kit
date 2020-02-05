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
INDEX = list(CONFIG["index"].keys())


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


def reduce_into_table(
    list_of_feature_paths,
    format_suffix=FORMAT_SUFFIX,
    config=CONFIG,
):
    evttab = {}
    for level in config['levels']:
        evttab[level] = []

    for feature_path in list_of_feature_paths:
        num_levels = 0
        with tarfile.open(feature_path, "r") as tarfin:
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
        ll = []
        for level_part in evttab[level]:
            ll.append(pd.DataFrame(level_part))
        level_df = pd.concat(ll, sort=False)
        evttab[level] = level_df.to_records(index=False)
    return evttab


def write(
    path,
    event_table,
    config=CONFIG,
    format_suffix=FORMAT_SUFFIX
):
    assert op.splitext(path)[1] == '.tar'
    with tarfile.open(path+".tmp", 'w') as tarout:
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
    shutil.move(path+".tmp", path)


def reduce(
    list_of_feature_paths,
    out_path,
    format_suffix=FORMAT_SUFFIX,
    config=CONFIG,
):
    event_table = reduce_into_table(
        list_of_feature_paths=list_of_feature_paths,
        format_suffix=format_suffix,
        config=config)
    write(
        path=out_path,
        event_table=event_table,
        config=config,
        format_suffix=format_suffix)


def read(
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


def find_common_indices(level_a, level_b):
    _a = {}
    _b = {}
    for idx in INDEX:
        _a[idx] = level_a[idx]
        _b[idx] = level_b[idx]
    merge_df = pd.merge(
        pd.DataFrame(_a),
        pd.DataFrame(_b),
        on=INDEX)[INDEX]
    return merge_df.to_records(index=False)


def mask_to_indices(level, mask):
    _part = {}
    for idx in INDEX:
        _part[idx] = level[idx]
    level_df = pd.DataFrame(_part)
    del _part
    level_mask_df = level_df[mask]
    return level_mask_df.to_records(index=False)


def by_indices(event_table, level_key, indices, keys=None):
    if keys == None:
        keys = CONFIG['levels'][level_key].keys()
    _part = {}
    for idx in INDEX:
        _part[idx] = event_table[level_key][idx]
    for key in keys:
        _part[key] = event_table[level_key][key]
    part_df = pd.DataFrame(_part)
    del _part
    common_df = pd.merge(
        part_df,
        pd.DataFrame(indices),
        on=INDEX,
        how='inner')
    del part_df
    common = common_df.to_records(index=False)
    del common_df
    common_order_args = np.argsort(_unique_index(common))
    common_sorted = common[common_order_args]
    del common_order_args
    indices_order_args = np.argsort(_unique_index(indices))
    inverse_order = np.zeros(shape=indices_order_args.shape, dtype=np.int)
    inverse_order[indices_order_args] = np.arange(len(indices))
    del indices_order_args
    common_same_order_as_indices = common_sorted[inverse_order]
    del inverse_order
    for idx in INDEX:
        np.testing.assert_array_equal(
            common_same_order_as_indices[idx],
            indices[idx])
    return common_same_order_as_indices


def merge(event_table, level_keys=CONFIG_LEVELS_KEYS):
    common = _find_common_indices(
        event_table=event_table,
        level_keys=level_keys)
    out = {}
    for idx in INDEX:
        out[idx] = common[idx]
    for level_key in level_keys:
        for key in CONFIG['levels'][level_key].keys():
            out['{:s}.{:s}'.format(level_key, key)] = by_indices(
                event_table=event_table,
                level_key=level_key,
                indices=common,
                keys=[key])[key]
    out_df = pd.DataFrame(out)
    del out
    return out_df.to_records(index=False)


UNIQUE_INDEX_FACTOR = (MAX_NUM_EVENTS_IN_RUN*10)


def _find_common_indices(event_table, level_keys=CONFIG_LEVELS_KEYS):
    uids = _unique_index(event_table[level_keys[0]])
    for lidx in np.arange(1, len(level_keys)):
        level_key = level_keys[lidx]
        _uids = _unique_index(event_table[level_key])
        uids = np.intersect1d(uids, _uids)
    run_ids = uids//UNIQUE_INDEX_FACTOR
    airshower_ids = uids%UNIQUE_INDEX_FACTOR
    df = pd.DataFrame({'run_id': run_ids, 'airshower_id': airshower_ids})
    return df.to_records(index=False)


def _unique_index(level):
     return level['run_id']*UNIQUE_INDEX_FACTOR + level['airshower_id']
