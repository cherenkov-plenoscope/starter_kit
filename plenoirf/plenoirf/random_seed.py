import numpy as np

NUM_DIGITS_RUN_ID = 6
NUM_DIGITS_AIRSHOWER_ID = 3
NUM_DIGITS_SEED = NUM_DIGITS_RUN_ID + NUM_DIGITS_AIRSHOWER_ID

NUM_AIRSHOWER_IDS_IN_RUN = 10 ** NUM_DIGITS_AIRSHOWER_ID
NUM_RUN_IDS = 10 ** NUM_DIGITS_RUN_ID
NUM_SEEDS = NUM_AIRSHOWER_IDS_IN_RUN * NUM_RUN_IDS

SEED_TEMPLATE_STR = "{seed:0" + str(NUM_DIGITS_SEED) + "d}"


def random_seed_based_on(run_id, airshower_id):
    assert is_valid_run_id(run_id)
    assert is_valid_airshower_id(airshower_id)
    return run_id * NUM_AIRSHOWER_IDS_IN_RUN + airshower_id


def run_id_from_seed(seed):
    if np.isscalar(seed):
        assert seed <= NUM_SEEDS
    else:
        seed = np.array(seed)
        assert (seed <= NUM_SEEDS).all()
    return seed // NUM_AIRSHOWER_IDS_IN_RUN


def airshower_id_from_seed(seed):
    return seed - run_id_from_seed(seed) * NUM_AIRSHOWER_IDS_IN_RUN


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
