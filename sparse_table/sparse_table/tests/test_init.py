import sparse_table as spt
import numpy as np
import pandas as pd
import tempfile
import os


EXAMPLE_TABLE_STRUCTURE = {
    'elementary_school': {
        'lunchpack_size': {'dtype': '<f8'},
        'num_friends': {'dtype': '<i8'},
    },
    'high_school': {
        'time_spent_on_homework': {'dtype': '<f8'},
        'num_best_friends': {'dtype': '<i8'},
    },
    'university': {
        'num_missed_classes': {'dtype': '<i8'},
        'num_fellow_students': {'dtype': '<i8'},
    },
}


def _make_example_table(size, start_index=0):
    """
    Children start in elementary school. 10% progress to high school, and 10%
    of those progress to university.
    At each point in their career statistics are collected that can be put to
    columns, while every child is represented by a line.
    Unfortunately, a typical example of a sparse table.
    """
    t = {}
    t['elementary_school'] = spt.dict_to_recarray(
        {
            spt.IDX: start_index + np.arange(size).astype(spt.IDX_DTYPE),
            'lunchpack_size': np.random.uniform(size=size).astype('<f8'),
            'num_friends': np.random.uniform(
                low=0,
                high=5,
                size=size).astype('<i8'),
        }
    )
    high_school_size = size//10
    t['high_school'] = spt.dict_to_recarray(
        {
            spt.IDX: np.random.choice(
                t['elementary_school'][spt.IDX],
                size=high_school_size,
                replace=False),
            'time_spent_on_homework': 100 + 100*np.random.uniform(
                size=high_school_size).astype('<f8'),
            'num_best_friends': np.random.uniform(
                low=0,
                high=5,
                size=high_school_size).astype('<i8'),
        }
    )
    university_size = high_school_size//10
    t['university'] = spt.dict_to_recarray(
        {
            spt.IDX: np.random.choice(
                t['high_school'][spt.IDX],
                size=university_size,
                replace=False),
            'num_missed_classes': 100*np.random.uniform(
                size=university_size).astype('<i8'),
            'num_fellow_students': np.random.uniform(
                low=0,
                high=5,
                size=university_size).astype('<i8'),
        }
    )
    spt.assert_structure_keys_are_valid(structure=EXAMPLE_TABLE_STRUCTURE)
    spt.assert_table_has_structure(table=t, structure=EXAMPLE_TABLE_STRUCTURE)
    return t


def test_from_records():
    np.random.seed(0)
    rnd = np.random.uniform

    # define what your table will look like
    # -------------------------------------
    structure = {
        "A": {
            "a": {"dtype": '<f8'},
            "b": {"dtype": '<f8'},
        },
        "B": {
            "c": {"dtype": '<f8'},
            "d": {"dtype": '<f8'},
        },
        "C": {
            "e": {"dtype": '<f8'},
        },
    }

    # populate the table using records
    # --------------------------------
    with tempfile.TemporaryDirectory(prefix='test_sparse_table') as tmp:

        num_jobs = 100
        n = 5
        job_result_paths = []
        for j in range(num_jobs):

            # map the population of the sparse table onto many jobs
            # -----------------------------------------------------
            i = j*n
            table_records = {}

            table_records["A"] = []
            table_records["A"].append({spt.IDX: i+0, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i+1, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i+2, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i+3, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i+4, "a": rnd(), "b": rnd()})

            table_records["B"] = []
            table_records["B"].append({spt.IDX: i+0, "c": rnd(), "d": 5*rnd()})
            table_records["B"].append({spt.IDX: i+3, "c": rnd(), "d": 5*rnd()})

            table_records["C"] = []
            if rnd() > 0.9:
                table_records["C"].append({spt.IDX: i+3, "e": -rnd()})

            table = spt.table_of_records_to_sparse_table(
                table_records=table_records,
                structure=structure)

            path = os.path.join(tmp, '{:06d}.tar'.format(j))
            job_result_paths.append(path)
            spt.write(path=path, table=table, structure=structure)

        # reduce
        # ------
        full_table = spt.concatenate_files(
            list_of_table_paths=job_result_paths,
            structure=structure)

    spt.assert_table_has_structure(
        table=full_table,
        structure=structure)


def test_write_read_full_table():
    np.random.seed(1337)
    my_table = _make_example_table(size=1000*1000)
    with tempfile.TemporaryDirectory(prefix='test_sparse_table') as tmp:
        path = os.path.join(tmp, 'my_table.tar')
        spt.write(
            path=path,
            table=my_table,
            structure=EXAMPLE_TABLE_STRUCTURE)
        my_table_back = spt.read(
            path=path,
            structure=EXAMPLE_TABLE_STRUCTURE)
        spt.assert_tables_are_equal(my_table, my_table_back)


def test_write_read_empty_table():
    np.random.seed(1337)
    empty_table = _make_example_table(size=0)
    with tempfile.TemporaryDirectory(prefix='test_sparse_table') as tmp:
        path = os.path.join(tmp, 'my_empty_table.tar')
        spt.write(
            path=path,
            table=empty_table,
            structure=EXAMPLE_TABLE_STRUCTURE)
        my_table_back = spt.read(
            path=path,
            structure=EXAMPLE_TABLE_STRUCTURE)
        spt.assert_tables_are_equal(empty_table, my_table_back)


def test_merge_common():
    np.random.seed(1337)
    my_table = _make_example_table(size=1000*1000)

    common_indices = spt.find_common_indices(
        table=my_table,
        structure=EXAMPLE_TABLE_STRUCTURE)

    my_common_table = spt.cut_table_on_indices(
        table=my_table,
        structure=EXAMPLE_TABLE_STRUCTURE,
        common_indices=common_indices)

    np.testing.assert_array_equal(
        my_common_table['elementary_school'][spt.IDX],
        my_common_table['high_school'][spt.IDX])

    np.testing.assert_array_equal(
        my_common_table['elementary_school'][spt.IDX],
        my_common_table['university'][spt.IDX])

    my_common_df = spt.make_rectangular_DataFrame(table=my_common_table)

    np.testing.assert_array_equal(
        my_common_table['elementary_school'][spt.IDX],
        my_common_df[spt.IDX])


def test_merge_across_all_levels_random_order_indices():
    np.random.seed(1337)
    size = 1000*1000
    my_table = _make_example_table(size=size)

    has_elementary_school = my_table['elementary_school'][spt.IDX]
    has_high_school = my_table['high_school'][spt.IDX]
    has_university = my_table['university'][spt.IDX]
    has_big_lunchpack = my_table['elementary_school'][spt.IDX][
        my_table['elementary_school']['lunchpack_size'] > 0.5]
    has_2best_friends = my_table['high_school'][spt.IDX][
        my_table['high_school']['num_best_friends'] >= 2]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_university)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = spt.cut_table_on_indices(
        table=my_table,
        structure=EXAMPLE_TABLE_STRUCTURE,
        common_indices=cut_indices,
        level_keys=['elementary_school', 'high_school', 'university'])

    np.testing.assert_array_equal(
        cut_table['elementary_school'][spt.IDX],
        cut_table['high_school'][spt.IDX])
    np.testing.assert_array_equal(
        cut_table['elementary_school'][spt.IDX],
        cut_table['university'][spt.IDX])
    np.testing.assert_array_equal(
        cut_table['elementary_school'][spt.IDX],
        cut_indices)


def test_merge_random_order_indices():
    np.random.seed(1337)
    size = 1000*1000
    my_table = _make_example_table(size=size)

    has_elementary_school = my_table['elementary_school'][spt.IDX]
    has_high_school = my_table['high_school'][spt.IDX]
    has_big_lunchpack = my_table['elementary_school'][spt.IDX][
        my_table['elementary_school']['lunchpack_size'] > 0.5]
    has_2best_friends = my_table['high_school'][spt.IDX][
        my_table['high_school']['num_best_friends'] >= 2]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = spt.cut_table_on_indices(
        table=my_table,
        structure=EXAMPLE_TABLE_STRUCTURE,
        common_indices=cut_indices,
        level_keys=['elementary_school', 'high_school'])

    assert 'university' not in cut_table
    assert 'elementary_school' in cut_table
    assert 'high_school' in cut_table

    np.testing.assert_array_equal(
        cut_table['elementary_school'][spt.IDX],
        cut_table['high_school'][spt.IDX])
    np.testing.assert_array_equal(
        cut_table['elementary_school'][spt.IDX],
        cut_indices)


def test_concatenate_several_tables():
    np.random.seed(1337)
    block_size = 10*1000
    num_blocks = 100

    with tempfile.TemporaryDirectory(prefix='test_sparse_table') as tmp:
        paths = []
        for i in range(num_blocks):
            table_i = _make_example_table(
                size=block_size,
                start_index=i*block_size)
            paths.append(os.path.join(tmp, "{:06d}.tar".format(i)))
            spt.write(
                path=paths[-1],
                table=table_i,
                structure=EXAMPLE_TABLE_STRUCTURE)
        output_path = os.path.join(tmp, "full.tar")
        full_table = spt.concatenate_files(
            list_of_table_paths=paths,
            structure=EXAMPLE_TABLE_STRUCTURE,)
    spt.assert_table_has_structure(
        table=full_table,
        structure=EXAMPLE_TABLE_STRUCTURE)

    assert (
        full_table['elementary_school'][spt.IDX].shape[0] ==
        num_blocks*block_size)
    assert (
        len(set(full_table['elementary_school'][spt.IDX])) ==
        num_blocks*block_size), "The indices must be uniqe"
    assert (
        full_table['high_school'][spt.IDX].shape[0] ==
        num_blocks*block_size//10)
    assert (
        len(set(full_table['high_school'][spt.IDX])) ==
        num_blocks*block_size//10)
    assert (
        full_table['university'][spt.IDX].shape[0] ==
        num_blocks*block_size//100)
    assert (
        len(set(full_table['university'][spt.IDX])) ==
        num_blocks*block_size//100)


def test_concatenate_empty_list_of_paths():
    with tempfile.TemporaryDirectory(prefix='test_sparse_table') as tmp:
        output_path = os.path.join(tmp, 'empty_table.tar')
        empty_table = spt.concatenate_files(
            list_of_table_paths=[],
            structure=EXAMPLE_TABLE_STRUCTURE)
    assert empty_table['elementary_school'][spt.IDX].shape[0] == 0


def test_only_index_in_level():
    np.random.seed(42)

    structure = {
        "A": {"height": {"dtype": "<i8"}},
        "B": {},
    }

    table = {}
    table["A"] = spt.dict_to_recarray({
        spt.IDX: np.arange(10).astype(spt.IDX_DTYPE),
        "height": np.ones(10, dtype='<i8'),
    })
    table["B"] = spt.dict_to_recarray({
        spt.IDX: np.random.choice(table["A"][spt.IDX], 5),
    })

    spt.assert_table_has_structure(
        table=table,
        structure=structure)

    with tempfile.TemporaryDirectory(prefix='test_sparse_table') as tmp:
        path = os.path.join(tmp, 'table_with_index_only_level.tar')
        spt.write(
            path=path,
            table=table,
            structure=structure)
        table_back = spt.read(path=path, structure=structure)
        spt.assert_tables_are_equal(table, table_back)
