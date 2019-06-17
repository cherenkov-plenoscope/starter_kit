import gev_limits as gli
import numpy as np
import tempfile
import os


MERLICT_PATH = os.path.abspath('./build/merlict/merlict-eventio-converter')


def test_find_merlict():
    assert os.path.exists(MERLICT_PATH)


def test_map_and_reduce():
    with tempfile.TemporaryDirectory(prefix='gev_limits_') as tmp:
        jobs = gli.map_and_reduce.make_jobs(
            map_and_reduce_dir=tmp,
            random_seed=1,
            num_runs=2,
            num_events_in_run=100,
            eventio_converter_path=MERLICT_PATH,
            instrument=gli.job_structure.example_job['instrument'],
            particle=gli.job_structure.example_job['particle'],
            site=gli.job_structure.example_job['site'],
            trigger_threshold=gli.job_structure.example_job['trigger_threshold'],
        )
        assert len(jobs) == 2

        for job in jobs:
            gli.map_and_reduce.run_job(job)

        assert os.path.exists(jobs[0]['out_path'])
        assert os.path.exists(jobs[1]['out_path'])

        all_runs_path = os.path.join(tmp, 'all_runs')
        gli.lookup.concatenate([
            jobs[0]['out_path'],
            jobs[1]['out_path']
            ], all_runs_path)

        lut = gli.lookup.LookUpTable(all_runs_path)
        assert lut.num_events > 0

        thrown = gli.thrown_structure.read_events_thrown(
            os.path.join(all_runs_path, 'thrown.float32'))

        assert thrown['particle_id'].shape[0] == 200
