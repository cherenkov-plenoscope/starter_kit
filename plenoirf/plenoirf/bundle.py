import numpy as np
import copy
import sys
from . import map_and_reduce


def bundle_jobs(jobs, desired_num_bunbles):
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
    return job * job


def _run_bundle_example(bundle):
    results = []
    for j, job in enumerate(bundle):
        result = _run_job_example(job=job)
        results.append(result)
    return results
