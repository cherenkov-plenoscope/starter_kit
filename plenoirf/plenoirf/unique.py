RUN_ID_NUM_DIGITS = 6
EVENT_ID_NUM_DIGITS = 6
UID_NUM_DIGITS = RUN_ID_NUM_DIGITS + EVENT_ID_NUM_DIGITS

RUN_ID_UPPER = 10 ** RUN_ID_NUM_DIGITS
EVENT_ID_UPPER = 10 ** EVENT_ID_NUM_DIGITS

UID_FOTMAT_STR = "{:0" + str(UID_NUM_DIGITS) + "d}"


def make_uid(run_id, event_id):
    assert 0 <= run_id < RUN_ID_UPPER
    assert 0 <= event_id < EVENT_ID_UPPER
    return RUN_ID_UPPER * run_id + event_id


def split_uid(udi):
    run_id = udi // RUN_ID_UPPER
    event_id = udi % RUN_ID_UPPER
    return run_id, event_id


def make_udi_str(run_id, event_id):
    uid = make_uid(run_id, event_id)
    return UID_FOTMAT_STR.format(uid)


def split_udi_str(s):
    uid = int(s)
    return split_uid(uid)
