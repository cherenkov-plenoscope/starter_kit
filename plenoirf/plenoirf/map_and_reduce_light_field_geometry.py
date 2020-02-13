import numpy as np
import subprocess
import os

def make_jobs(
    merlict_map_path,
    scenery_path,
    num_photons_per_block,
    out_dir,
    num_blocks,
    random_seed=0
):
    jobs = []
    for seed in np.arange(random_seed, num_blocks):
        jobs.append({
            "merlict_map_path": merlict_map_path,
            "scenery_path": scenery_path,
            "random_seed": seed,
            "out_dir": out_dir,
            "num_photons_per_block": num_photons_per_block})
    return jobs


def run_job(job):
    seed_str = '{:d}'.format(job['random_seed'])
    call = [
        job['merlict_map_path'],
        '-s', job['scenery_path'],
        '-n', '{:d}'.format(job['num_photons_per_block']),
        '-o', os.path.join(job['out_dir'], seed_str),
        '-r', seed_str]
    return subprocess.call(call)
