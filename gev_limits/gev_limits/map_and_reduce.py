import numpy as np
import os
import tempfile
import shutil
from . import job_structure
from . import process_run
import acp_instrument_response_function as acpirf


def make_jobs(
    map_and_reduce_dir,
    random_seed,
    max_num_events_in_run,
    num_runs,
    instrument,
    particle,
    site,
    trigger_threshold,
    nsb_rate_pixel,
    eventio_converter_path='./build/merlict/merlict-eventio-converter',
):
    (
        max_scatter_radii,
        energy_bin_edges
    ) = acpirf.utils.energy_bins_and_max_scatter_radius(
            energy=particle['energy'],
            max_scatter_radius=particle['max_scatter_radius'],
            number_runs=num_runs)

    max_energy = particle['energy'][-1]

    map_and_reduce_dir = os.path.abspath(map_and_reduce_dir)
    eventio_converter_path = os.path.abspath(
        eventio_converter_path)
    jobs = []
    job_idx = 0
    for i in range(num_runs):
        run_particle = {
            'prmpar': particle['prmpar'],
            'max_zenith_angle_deg': particle['max_zenith_angle_deg'],
            'E_start': energy_bin_edges[i],
            'E_stop': energy_bin_edges[i + 1],
            'E_slope': -1.,
            'max_scatter_radius': max_scatter_radii[i]
        }

        num_events = max_num_events_in_run
        if num_events*run_particle['E_stop'] > 10*max_energy:
            num_events = int((10*max_energy)//run_particle['E_stop'])

        multiplicity_jobs = int(np.ceil(max_num_events_in_run/num_events))

        for m in range(multiplicity_jobs):
            job = {}
            job['random_seed'] = int(random_seed + job_idx)
            job['trigger_threshold'] = float(trigger_threshold)
            job['nsb_rate_pixel'] = float(nsb_rate_pixel)
            job['site'] = site
            job['particle'] = run_particle
            job['instrument'] = instrument
            job['eventio_converter_path'] = eventio_converter_path
            job['num_events'] = num_events
            job['out_path'] = os.path.join(
                map_and_reduce_dir,
                'run_{:06d}.lut'.format(job['random_seed']))
            jobs.append(job)
            job_idx += 1
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
            trigger_threshold=job['trigger_threshold'],
            nsb_rate_pixel=job['nsb_rate_pixel'])
        shutil.copytree(
            src=os.path.join(tmp, 'run_{:06d}.lut'.format(job['random_seed'])),
            dst=job['out_path'])
