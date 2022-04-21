import plenoirf
import numpy as np


def _flatten_bundles(bundles):
    jobs_back = []
    for b in bundles:
        jobs_back += b
    return jobs_back


def test_zero_jobs():
    bundles = plenoirf.bundle.make_jobs_in_bundles(
        jobs=[], desired_num_bunbles=1
    )
    assert len(bundles) == 0


def test_many_jobs_one_bundle():
    jobs = np.arange(1000).tolist()
    bundles = plenoirf.bundle.make_jobs_in_bundles(
        jobs=jobs, desired_num_bunbles=1
    )
    assert len(bundles) == 1

    jobs_back = _flatten_bundles(bundles)

    for j in range(len(jobs)):
        assert jobs_back[j] == jobs[j]


def test_many_jobs_many_bundles():
    jobs = np.arange(1000).tolist()
    bundles = plenoirf.bundle.make_jobs_in_bundles(
        jobs=jobs, desired_num_bunbles=10
    )
    assert len(bundles) == 10

    jobs_back = _flatten_bundles(bundles)

    for j in range(len(jobs)):
        assert jobs_back[j] == jobs[j]


def test_few_jobs_many_bundles():
    jobs = np.arange(10).tolist()
    bundles = plenoirf.bundle.make_jobs_in_bundles(
        jobs=jobs, desired_num_bunbles=100
    )
    assert len(bundles) == 10

    jobs_back = _flatten_bundles(bundles)

    for j in range(len(jobs)):
        assert jobs_back[j] == jobs[j]


def test_run_jobs_in_bundles():
    jobs = np.arange(24).tolist()
    bundles = plenoirf.bundle.make_jobs_in_bundles(
        jobs=jobs, desired_num_bunbles=3
    )

    bundles_results = []
    for bundle in bundles:
        bundle_results = plenoirf.bundle._run_jobs_in_bundles_example(
            bundle=bundle
        )
        bundles_results.append(bundle_results)

    job_results = _flatten_bundles(bundles_results)

    for j in range(len(jobs)):
        assert job_results[j] == plenoirf.bundle._run_job_example(jobs[j])
