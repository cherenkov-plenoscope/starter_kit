import numpy as np
import copy


def make_jobs_in_bundles(jobs, desired_num_bunbles):
    """
    When you have too many jobs for your parallel processing queue this
    function bundle multiple of your jobs into fewer bundles of jobs.

    Parameters
    ----------
    jobs : list
        A list of your jobs.
    desired_num_bunbles : int
        The maximum number of bundles. Your jobs will be spread over
        these many bundles.

    Returns
    -------
        A list of bundles where each bundle is a list of jobs.
        The lengths of the list of bundles is <= desired_num_bunbles.
    """
    assert desired_num_bunbles > 0
    bundles = []
    num_jobs = len(jobs)
    num_jobs_in_bundle = int(np.ceil(num_jobs / desired_num_bunbles))
    current_bundle = []
    for j in range(num_jobs):
        if len(current_bundle) < num_jobs_in_bundle:
            current_bundle.append(jobs[j])
        else:
            bundles.append(copy.deepcopy(current_bundle))
            current_bundle = []
            current_bundle.append(jobs[j])
    if len(current_bundle):
        bundles.append(current_bundle)
    return bundles


def _run_job_example(job):
    """
    For testing. This function processes one job.
    You already have this for your jobs.
    """
    return job * job


def _run_jobs_in_bundles_example(bundle):
    """
    For testing. This function processes one bundle of jobs by
    looping over the jobs in a bundle.
    You need to provide this function.
    """
    results = []
    for j, job in enumerate(bundle):
        result = _run_job_example(job=job)
        results.append(result)
    return results
