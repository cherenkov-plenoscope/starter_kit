
import plenoirf
import numpy as np
import pytest


def test_valid_airshower_id():
    assert not plenoirf.random_seed.is_valid_airshower_id(-1)
    assert plenoirf.random_seed.is_valid_airshower_id(0)
    assert plenoirf.random_seed.is_valid_airshower_id(1)

    assert plenoirf.random_seed.is_valid_airshower_id(
        plenoirf.random_seed.NUM_AIRSHOWER_IDS_IN_RUN - 1)
    assert not plenoirf.random_seed.is_valid_airshower_id(
        plenoirf.random_seed.NUM_AIRSHOWER_IDS_IN_RUN)


def test_valid_run_id():
    assert not plenoirf.random_seed.is_valid_run_id(-1)
    assert plenoirf.random_seed.is_valid_run_id(0)
    assert plenoirf.random_seed.is_valid_run_id(1)

    assert plenoirf.random_seed.is_valid_run_id(
        plenoirf.random_seed.NUM_RUN_IDS - 1)
    assert not plenoirf.random_seed.is_valid_run_id(
        plenoirf.random_seed.NUM_RUN_IDS)


def test_seed_limit_run_id():
    seed = plenoirf.random_seed.random_seed_based_on(
        run_id=plenoirf.random_seed.NUM_RUN_IDS - 1,
        airshower_id=1)
    assert (
        plenoirf.random_seed.NUM_RUN_IDS - 1 ==
        plenoirf.random_seed.run_id_from_seed(seed=seed)
    )
    assert 1 == plenoirf.random_seed.airshower_id_from_seed(seed=seed)

    with pytest.raises(AssertionError) as e:
        plenoirf.random_seed.random_seed_based_on(
            run_id=plenoirf.random_seed.NUM_RUN_IDS,
            airshower_id=1)


def test_seed_limit_airshower_id():
    seed = plenoirf.random_seed.random_seed_based_on(
        run_id=1,
        airshower_id=plenoirf.random_seed.NUM_AIRSHOWER_IDS_IN_RUN - 1)
    assert (
        plenoirf.random_seed.NUM_AIRSHOWER_IDS_IN_RUN - 1 ==
        plenoirf.random_seed.airshower_id_from_seed(seed=seed)
    )
    assert 1 == plenoirf.random_seed.run_id_from_seed(seed=seed)

    with pytest.raises(AssertionError) as e:
        plenoirf.random_seed.random_seed_based_on(
            run_id=0,
            airshower_id=plenoirf.random_seed.NUM_AIRSHOWER_IDS_IN_RUN)


def test_seed_combinations():
    np.random.seed(0)
    run_ids = np.random.uniform(
        0,
        plenoirf.random_seed.NUM_RUN_IDS - 1,
        size=300).astype('i4')
    airshower_ids = np.random.uniform(
        0,
        plenoirf.random_seed.NUM_AIRSHOWER_IDS_IN_RUN - 1,
        size=300).astype('i4')

    for run_id in run_ids:
        for airshower_id in airshower_ids:
            seed = plenoirf.random_seed.random_seed_based_on(
                run_id=run_id,
                airshower_id=airshower_id)
            np.random.seed(seed)
            assert plenoirf.random_seed.run_id_from_seed(seed=seed) == run_id
            assert (
                plenoirf.random_seed.airshower_id_from_seed(seed=seed) ==
                airshower_id
            )


def test_seed_num_digits():
    assert plenoirf.random_seed.NUM_DIGITS_SEED >= 6
    assert plenoirf.random_seed.NUM_DIGITS_SEED <= 12
    assert plenoirf.random_seed.NUM_SEEDS < np.iinfo(np.int32).max


def test_template_string():
    np.random.seed(0)
    run_ids = np.random.uniform(
        0,
        plenoirf.random_seed.NUM_RUN_IDS - 1,
        size=30).astype('i4')
    airshower_ids = np.random.uniform(
        0,
        plenoirf.random_seed.NUM_AIRSHOWER_IDS_IN_RUN - 1,
        size=30).astype('i4')

    for run_id in run_ids:
        for airshower_id in airshower_ids:
            seed = plenoirf.random_seed.random_seed_based_on(
                run_id=run_id,
                airshower_id=airshower_id)
            s = plenoirf.random_seed.SEED_TEMPLATE_STR.format(seed=seed)
            assert int(s) == seed
            assert len(s) == plenoirf.random_seed.NUM_DIGITS_SEED
