import gev_limits as gli
import numpy as np
import tempfile
import os


MERLICT_PATH = os.path.abspath('./build/merlict/merlict-eventio-converter')


def test_find_merlict():
    assert os.path.exists(MERLICT_PATH)


def test_runs():
    with tempfile.TemporaryDirectory(prefix='gev_limits_') as tmp:
        run1_path = os.path.join(tmp, 'run1')
        gli.process_run.process_run(
            tmp_dir=run1_path,
            random_seed=1,
            num_events=100,
            eventio_converter_path=MERLICT_PATH,
            instrument=gli.job_structure.example_job['instrument'],
            particle=gli.job_structure.example_job['particle'],
            site=gli.job_structure.example_job['site'],
            trigger_threshold=gli.job_structure.example_job['trigger_threshold'],
            nsb_rate_pixel=gli.job_structure.example_job['nsb_rate_pixel'],)

        run2_path = os.path.join(tmp, 'run2')
        gli.process_run.process_run(
            tmp_dir=run2_path,
            random_seed=2,
            num_events=100,
            eventio_converter_path=MERLICT_PATH,
            instrument=gli.job_structure.example_job['instrument'],
            particle=gli.job_structure.example_job['particle'],
            site=gli.job_structure.example_job['site'],
            trigger_threshold=gli.job_structure.example_job['trigger_threshold'],
            nsb_rate_pixel=gli.job_structure.example_job['nsb_rate_pixel'],)

        all_runs_path = os.path.join(tmp, 'all_runs')
        gli.lookup.concatenate([
            os.path.join(run1_path, 'run_{:06d}.lut'.format(1)),
            os.path.join(run2_path, 'run_{:06d}.lut'.format(2))
            ], all_runs_path)

        lut = gli.lookup.LookUpTable(all_runs_path)
        assert lut.num_events > 0

        thrown = gli.thrown_structure.read_events_thrown(
            os.path.join(all_runs_path, 'thrown.float32'))

        assert thrown['particle_id'].shape[0] == 200
