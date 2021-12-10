
def make_shower_id(run_id, event_id):
    assert 0 <= run_id < 1000*1000
    assert 0 <= event_id < 1000*1000
    return 1000*1000 * run_id + event_id


def split_shower_id(shower_id):
    run_id = shower_id // (1000*1000)
    event_id = shower_id % (1000*1000) 
    return run_id, event_id


def make_shower_id_str(run_id, event_id):
    assert 0 <= run_id < 1000*1000
    assert 0 <= event_id < 1000*1000
    return "{:06d}{:06d}".format(run_id, event_id)


def split_shower_id_str(s):
    run_id = int(s[0:6])
    event_id = int(s[6:12])
    return run_id, event_id
