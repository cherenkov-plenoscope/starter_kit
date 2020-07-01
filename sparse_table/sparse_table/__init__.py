import pandas as pd
import numpy as np
import tarfile
import io
import shutil
import tempfile
import os


"""
sparse tables
=============

    Might look like this:

        level 1          level 2       level 2
          columns          columns       columns
     idx a b c d e f  idx g h i j k l  idx m n o p
     ___ _ _ _ _ _ _  ___ _ _ _ _ _ _
    |_0_|_|_|_|_|_|_||_0_|_|_|_|_|_|_|
    |_1_|_|_|_|_|_|_|
    |_2_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |_3_|_|_|_|_|_|_||_3_|_|_|_|_|_|_|
    |_4_|_|_|_|_|_|_||_4_|_|_|_|_|_|_| ___ _ _ _ _
    |_5_|_|_|_|_|_|_||_5_|_|_|_|_|_|_||_5_|_|_|_|_|
    |_6_|_|_|_|_|_|_|
    |_7_|_|_|_|_|_|_|
    |_8_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |_9_|_|_|_|_|_|_||_9_|_|_|_|_|_|_|
    |10_|_|_|_|_|_|_||10_|_|_|_|_|_|_|
    |11_|_|_|_|_|_|_| ___ _ _ _ _ _ _  ___ _ _ _ _
    |12_|_|_|_|_|_|_||12_|_|_|_|_|_|_||12_|_|_|_|_|
    |13_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |14_|_|_|_|_|_|_||14_|_|_|_|_|_|_|


    Can be represented in memory like this:

        level 1            level 2         level 2
          columns            columns         columns
     idx a b c d e f    idx g h i j k l    idx m n o p
     ___ _ _ _ _ _ _    ___ _ _ _ _ _ _    ___ _ _ _ _
    |_0_|_|_|_|_|_|_|  |_0_|_|_|_|_|_|_|  |_5_|_|_|_|_|
    |_1_|_|_|_|_|_|_|  |_3_|_|_|_|_|_|_|  |12_|_|_|_|_|
    |_2_|_|_|_|_|_|_|  |_4_|_|_|_|_|_|_|
    |_3_|_|_|_|_|_|_|  |_9_|_|_|_|_|_|_|
    |_4_|_|_|_|_|_|_|  |10_|_|_|_|_|_|_|
    |_6_|_|_|_|_|_|_|  |12_|_|_|_|_|_|_|
    |_7_|_|_|_|_|_|_|  |14_|_|_|_|_|_|_|
    |_8_|_|_|_|_|_|_|
    |_9_|_|_|_|_|_|_|
    |10_|_|_|_|_|_|_|
    |11_|_|_|_|_|_|_|
    |12_|_|_|_|_|_|_|
    |13_|_|_|_|_|_|_|
    |14_|_|_|_|_|_|_|

    Written to tape-archive

    table.tar
        |_ level_1;idx
        |_ level_1;column_a
        |_ level_1;column_b
        |_ level_1;column_c
        |_ level_1;column_d
        |_ level_1;column_e
        |_ level_1;column_f
        |_ level_2;idx
        |_ level_2;column_g
        |_ level_2;column_h
        |_ level_2;column_i
        |_ level_2;column_j
        |_ level_2;column_k
        |_ level_2;column_l
        |_ level_3;idx
        |_ level_3;column_m
        |_ level_3;column_n
        |_ level_3;column_o
        |_ level_3;column_p
"""


IDX = 'idx'
IDX_DTYPE = '<u8'

LEVEL_COLUMN_DELIMITER = ';'
FILEAME_TEMPLATE = '{:s}'+LEVEL_COLUMN_DELIMITER+'{:s}.{:s}'

DTYPES = [
    '<u1', '<u2', '<u4', '<u8',
    '<i1', '<i2', '<i4', '<i8',
    '<f2', '<f4', '<f8',
]

# logical operations
# ==================


def find_common_indices(table, structure, level_keys=None):
    if level_keys is None:
        level_keys = list(structure.keys())
    uids = table[level_keys[0]][IDX]
    for lidx in np.arange(1, len(level_keys)):
        level_key = level_keys[lidx]
        _uids = table[level_key][IDX]
        uids = np.intersect1d(uids, _uids)
    return uids


def intersection(list_of_lists_of_indices):
    inter = list_of_lists_of_indices[0]
    for i in range(len(list_of_lists_of_indices)):
        inter = np.intersect1d(inter, list_of_lists_of_indices[i])
    return inter


def cut_level_on_indices(
    table,
    structure,
    level_key,
    indices,
    column_keys=None
):
    """
    Returns level 'level_key' of 'table' only containing 'indices' while
    having the same order as 'indices'.

    indices                         Recarray, with indices

    column_keys                     A list of column_keys to be in the output
                                    level. Uese all column_keys in structure
                                    for column_keys == None.
    """
    if column_keys is None:
        column_keys = list(structure[level_key].keys())
    column_keys.append(IDX)
    _part = {}
    for column_key in column_keys:
        _part[column_key] = table[level_key][column_key]
    part_df = pd.DataFrame(_part)
    del _part
    common_df = pd.merge(
        part_df,
        pd.DataFrame(dict_to_recarray({IDX: indices})),
        on=IDX,
        how='inner')
    del part_df
    return common_df.to_records(index=False)


def cut_table_on_indices(
    table,
    structure,
    common_indices,
    level_keys=None
):
    if level_keys is None:
        level_keys = list(structure.keys())
    out = {}
    for level_key in level_keys:
        out[level_key] = cut_level_on_indices(
            table=table,
            structure=structure,
            level_key=level_key,
            indices=common_indices)
    return out


def sort_table_on_common_indices(
    table,
    common_indices,
):
    common_order_args = np.argsort(common_indices)
    common_inv_order = np.zeros(shape=common_indices.shape, dtype=np.int)
    common_inv_order[common_order_args] = np.arange(len(common_indices))
    del common_order_args

    out = {}
    for level_key in table:
        level = table[level_key]
        level_order_args = np.argsort(level[IDX])
        level_sorted = level[level_order_args]
        del level_order_args
        level_same_order_as_common = level_sorted[common_inv_order]
        out[level_key] = level_same_order_as_common
    return out


def make_mask_of_right_in_left(left_indices, right_indices):
    left_df = pd.DataFrame({IDX: left_indices})
    right_df = pd.DataFrame({IDX: right_indices})
    mask_df = pd.merge(left_df, right_df, on=IDX, how='left', indicator=True)
    indicator_df = mask_df['_merge']
    mask = np.array(indicator_df == 'both', dtype=np.int64)
    return mask


def make_rectangular_DataFrame(table):
    out = {}
    for level_key in table:
        for column_key in table[level_key].dtype.names:
            if column_key == IDX:
                if IDX in out:
                    np.testing.assert_array_equal(
                        out[IDX],
                        table[level_key][IDX])
                else:
                    out[IDX] = table[level_key][IDX]
            else:
                out["{:s}.{:s}".format(level_key, column_key)] = table[
                    level_key][column_key]
    return pd.DataFrame(out)


# assertion
# =========

def _assert_same_keys(keys_a, keys_b):
    uni_keys = list(set(keys_a + keys_b))
    for key in uni_keys:
        assert key in keys_a and key in keys_b, 'Key: {:s}'.format(key)


def assert_tables_are_equal(table_a, table_b):
    _assert_same_keys(list(table_a.keys()), list(table_b.keys()))
    for level_key in table_a:
        _assert_same_keys(
            table_a[level_key].dtype.names,
            table_b[level_key].dtype.names)
        for column_key in table_a[level_key].dtype.names:
            assert (
                table_a[level_key].dtype[column_key] ==
                table_b[level_key].dtype[column_key])
            np.testing.assert_array_equal(
                x=table_a[level_key][column_key],
                y=table_b[level_key][column_key],
                err_msg='table[{:s}][{:s}]'.format(level_key, column_key),
                verbose=True)


def assert_table_has_structure(table, structure):
    for level_key in structure:
        assert level_key in table, (
            "Expected level '{:s}' in table, but it is not.".format(level_key))
        assert IDX in table[level_key].dtype.names, (
            "Expected table[{:s}] to have column '{:s}', "
            "but it has not.".format(level_key, IDX))
        assert IDX_DTYPE == table[level_key].dtype[IDX], (
            "Expected table[{:s}][{:s}].dtype == {:s}"
            "but actually it is {:s}.".format(
                level_key,
                IDX,
                str(IDX_DTYPE),
                str(table[level_key].dtype[IDX]))
            )
        for column_key in structure[level_key]:
            assert column_key in table[level_key].dtype.names, (
                "Expected column '{:s}' in table's level '{:s}', "
                "but it is not.".format(
                    column_key,
                    level_key)
            )
            expected_dtype = structure[level_key][column_key]['dtype']
            assert expected_dtype == table[level_key].dtype[column_key], (
                "Expected table[{level_key:s}][{column_key:s}].dtype "
                "== {expected_dtype:s}, "
                "but actually it is {actual_dtype:s}".format(
                    level_key=level_key,
                    column_key=column_key,
                    expected_dtype=str(expected_dtype),
                    actual_dtype=str(table[level_key].dtype[column_key]))
                )


def dict_to_recarray(d):
    return pd.DataFrame(d).to_records(index=False)


def _assert_no_whitespace(key):
    for char in key:
        assert not str.isspace(char), (
            "Key must not contain spaces, but key = '{:s}'".format(key))


def _assert_no_dot(key):
    assert '.' not in key, (
        "Key must not contain '.', but key = '{:s}'".format(key))


def _assert_no_directory_delimeter(key):
    assert '/' not in key, (
        "Key must not contain '/', but key = '{:s}'".format(key))
    assert '\\' not in key, (
        "Key must not contain '\\', but key = '{:s}'".format(key))


def _assert_key_is_valid(key):
    _assert_no_whitespace(key)
    _assert_no_dot(key)
    _assert_no_directory_delimeter(key)


def assert_structure_keys_are_valid(structure):
    for level_key in structure:
        _assert_key_is_valid(level_key)
        for column_key in structure[level_key]:
            assert IDX != column_key
            _assert_key_is_valid(column_key)
            assert structure[level_key][column_key]['dtype'] in DTYPES, (
                "Structure[{:s}][{:s}]['dtype'] = {:s} "
                "is not in DTYPES".format(
                    level_key,
                    column_key,
                    str(structure[level_key][column_key]['dtype'])
                )
            )

# input output
# ============


def _append_tar(tarfout, name, payload_bytes):
    tarinfo = tarfile.TarInfo()
    tarinfo.name = name
    tarinfo.size = len(payload_bytes)
    with io.BytesIO() as fileobj:
        fileobj.write(payload_bytes)
        fileobj.seek(0)
        tarfout.addfile(tarinfo=tarinfo, fileobj=fileobj)


def write(path, table, structure):
    assert_table_has_structure(table=table, structure=structure)
    with tarfile.open(path+'.tmp', 'w') as tarfout:
        for level_key in structure:
            _append_tar(
                tarfout=tarfout,
                name=FILEAME_TEMPLATE.format(level_key, IDX, IDX_DTYPE),
                payload_bytes=table[level_key][IDX].tobytes())
            for column_key in structure[level_key]:
                dtype = structure[level_key][column_key]['dtype']
                _append_tar(
                    tarfout=tarfout,
                    name=FILEAME_TEMPLATE.format(level_key, column_key, dtype),
                    payload_bytes=table[level_key][column_key].tobytes())
    shutil.move(path+".tmp", path)


def read(path, structure):
    """
    Returns table and structure
    """
    out = {}
    for level_key in structure:
        out[level_key] = {}
    with tarfile.open(path, 'r') as tarfin:
        for tarinfo in tarfin:
            level_key, column_key_and_dtype = str.split(
                tarinfo.name,
                LEVEL_COLUMN_DELIMITER)
            column_key, dtype = str.split(column_key_and_dtype, '.')
            if column_key == IDX:
                assert dtype == IDX_DTYPE
            level_column_bytes = tarfin.extractfile(tarinfo).read()
            if level_key not in out:
                out[level_key] = {}
            out[level_key][column_key] = np.frombuffer(
                level_column_bytes,
                dtype=dtype)
    for level_key in out:
        out[level_key] = dict_to_recarray(out[level_key])
    assert_table_has_structure(table=out, structure=structure)
    return out


# concatenate
# ===========


def _make_tmp_paths(tmp, structure):
    tmp_paths = {}
    for level_key in structure:
        tmp_paths[level_key] = {}
        idx_fname = FILEAME_TEMPLATE.format(level_key, IDX, IDX_DTYPE)
        tmp_paths[level_key][IDX] = os.path.join(tmp, idx_fname)
        for column_key in structure[level_key]:
            col_dt = structure[level_key][column_key]['dtype']
            col_fname = FILEAME_TEMPLATE.format(level_key, column_key, col_dt)
            tmp_paths[level_key][column_key] = os.path.join(tmp, col_fname)
    return tmp_paths


def concatenate_files(
    list_of_table_paths,
    structure,
):
    with tempfile.TemporaryDirectory(prefix='sparse_table_concatenate') as tmp:
        tmp_paths = _make_tmp_paths(tmp=tmp, structure=structure)
        for table_path in list_of_table_paths:
            _part_table = read(path=table_path, structure=structure)
            for level_key in tmp_paths:
                for column_key in tmp_paths[level_key]:
                    with open(tmp_paths[level_key][column_key], "ab") as fa:
                        fa.write(_part_table[level_key][column_key].tobytes())
        out = {}
        for level_key in tmp_paths:
            out[level_key] = {}
            for column_key in tmp_paths[level_key]:
                if column_key == IDX:
                    dtype = IDX_DTYPE
                else:
                    dtype = structure[level_key][column_key]['dtype']

                if os.path.exists(tmp_paths[level_key][column_key]):
                    out[level_key][column_key] = np.fromfile(
                        tmp_paths[level_key][column_key],
                        dtype=dtype)
                else:
                    out[level_key][column_key] = np.zeros(0, dtype=dtype)
    for level_key in out:
        out[level_key] = dict_to_recarray(out[level_key])
    return out

# from records
# ============


def _empty_recarray(structure, level_key):
    dd = {IDX: np.zeros(0, dtype=IDX_DTYPE)}
    for column_key in structure[level_key]:
        dd[column_key] = np.zeros(
            0,
            dtype=structure[level_key][column_key]['dtype'])
    return dict_to_recarray(dd)


def records_to_recarray(level_records, level_key, structure):
    expected_keys = list(structure[level_key].keys()) + [IDX]
    if len(level_records) > 0:
        for record in level_records:
            _assert_same_keys(expected_keys, list(record.keys()))
        df = pd.DataFrame(level_records)
        dd = {IDX: df[IDX].values.astype(IDX_DTYPE)}
        for column_key in structure[level_key]:
            dd[column_key] = df[column_key].values.astype(
                structure[level_key][column_key]['dtype'])
        return dict_to_recarray(dd)
    else:
        return _empty_recarray(structure=structure, level_key=level_key)


def table_of_records_to_sparse_table(table_records, structure):
    table = {}
    for level_key in table_records:
        table[level_key] = records_to_recarray(
            level_records=table_records[level_key],
            level_key=level_key,
            structure=structure)
    return table
