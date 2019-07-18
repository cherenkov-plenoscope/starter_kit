import plenoscope_map_reduce as plmr
import numpy as np


def test_split_list_into_list_of_lists_empty():
    events = []
    jobs = plmr.split_list_into_list_of_lists(
        events=events,
        num_events_in_job=100)
    assert len(jobs) == 0


def test_split_list_into_list_of_lists_example():
    events = list(np.arange(1001))
    jobs = plmr.split_list_into_list_of_lists(
        events=events,
        num_events_in_job=100)

    assert len(jobs) == 11
    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        assert len(jobs[j]) == 100
    assert len(jobs[10]) == 1

    flat = []
    for job in jobs:
        for event in job:
            flat.append(event)
    assert len(flat) == len(events)
    assert sorted(flat) == sorted(events)


def test_split_list_into_list_of_lists_combinations():
    for num_events in np.arange(1000, 1100, 5):
        for num_events_in_job in np.arange(11, 37, 3):
            events = list(np.arange(num_events))
            jobs = plmr.split_list_into_list_of_lists(
                events=events,
                num_events_in_job=num_events_in_job)

            expected_num_jobs = int(np.ceil(num_events/num_events_in_job))
            expected_num_events_in_last_job = np.mod(
                num_events,
                num_events_in_job)

            assert len(jobs) == expected_num_jobs
            if expected_num_events_in_last_job == 0:
                for j in range(expected_num_jobs):
                    assert len(jobs[j]) == num_events_in_job
            else:
                for j in range(expected_num_jobs - 1):
                    assert len(jobs[j]) == num_events_in_job
                last_job_idx = expected_num_jobs - 1
                assert (
                    len(jobs[last_job_idx]) ==
                    expected_num_events_in_last_job)

            flat = []
            for job in jobs:
                for event in job:
                    flat.append(event)
            assert len(flat) == len(events)
            assert sorted(flat) == sorted(events)