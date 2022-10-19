import numpy as np
import subprocess
import os


def make_jobs(
    merlict_map_path,
    scenery_path,
    num_photons_per_block,
    map_dir,
    num_blocks,
    random_seed=0,
):
    """
    Returns a list of jobs (dicts) which can be processed by run_job().
    Each job adds a bit more statistics to the estimate of the light-field's
    geometry.

    Parameters
    ----------
    merlict_map_path : str
        Path to the executable. In merlict, executing one job.
    scenery_path : str
        Path to the scenery containing the instrument of which the
        light-field's geometry will be estimated of.
    num_photons_per_block: int
        The number of photons to be thrown in a single job.
    map_dir : str
        Path to the directory where the mapping of the jobs is done, i.e. where
        the jobs write their individual output to.
    num_blocks : int
        The number of jobs.
    random_seed : int
        The random_seed for the estimate.
    """
    jobs = []
    running_seed = int(random_seed)

    for i in range(num_blocks):
        jobs.append(
            {
                "merlict_map_path": merlict_map_path,
                "scenery_path": scenery_path,
                "random_seed": running_seed,
                "map_dir": map_dir,
                "num_photons_per_block": num_photons_per_block,
            }
        )
        running_seed += 1
    return jobs


def run_job(job):
    """
    Wrap the merlict executable to run a job for the parallel estimate
    of the light-field's geometry.

    Parameters
    ----------
    job : dict
    """
    seed_str = "{:d}".format(job["random_seed"])
    call = [
        job["merlict_map_path"],
        "-s",
        job["scenery_path"],
        "-n",
        "{:d}".format(job["num_photons_per_block"]),
        "-o",
        os.path.join(job["map_dir"], seed_str),
        "-r",
        seed_str,
    ]
    return subprocess.call(call)


def reduce(merlict_reduce_path, map_dir, out_dir):
    """
    Reduce the intermediate results in the map_dir and write them to the
    out_dir.

    Parameters
    ----------
    merlict_reduce_path : str
        Path to the executable. In merlict, reducing the results of the jobs.
    map_dir : str
        Path to the directory where jobs have written their output to.
    out_dir : str
        Path to the output directory which will represent the
        light-field-geometry.
    """
    return subprocess.call(
        [merlict_reduce_path, "--input", map_dir, "--output", out_dir,]
    )
