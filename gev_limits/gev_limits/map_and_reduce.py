import numpy as np
import os
import tempfile
import shutil
from . import job_structure
from . import process_run


def make_jobs(
    map_and_reduce_dir,
    random_seed=1,
    num_events_in_run=100,
    num_runs=100,
    eventio_converter_path='./build/merlict/merlict-eventio-converter',
    instrument=job_structure.example_job['instrument'],
    particle=job_structure.example_job['particle'],
    site=job_structure.example_job['site'],
    trigger_threshold=job_structure.example_job['trigger_threshold'],
):
    map_and_reduce_dir = os.path.abspath(map_and_reduce_dir)
    eventio_converter_path = os.path.abspath(
        eventio_converter_path)
    jobs = []
    for job_idx in range(num_runs):
        job = {}
        job['random_seed'] = int(random_seed + job_idx)
        job['trigger_threshold'] = float(trigger_threshold)
        job['site'] = site
        job['particle'] = particle
        job['instrument'] = instrument
        job['eventio_converter_path'] = eventio_converter_path
        job['num_events'] = num_events_in_run
        job['out_path'] = os.path.join(
            map_and_reduce_dir,
            'run_{:06d}.lut'.format(job['random_seed']))
        jobs.append(job)
    return jobs


def run_job(job):
    with tempfile.TemporaryDirectory(prefix='gev_limits_') as tmp:
        process_run.process_run(
            tmp_dir=tmp,
            random_seed=job['random_seed'],
            num_events=job['num_events'],
            eventio_converter_path=job['eventio_converter_path'],
            instrument=job['instrument'],
            particle=job['particle'],
            site=job['site'],
            trigger_threshold=job['trigger_threshold'])
        shutil.copytree(
            src=os.path.join(tmp, 'run_{:06d}.lut'.format(job['random_seed'])),
            dst=job['out_path'])
