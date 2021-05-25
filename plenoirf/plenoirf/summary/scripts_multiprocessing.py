import pkg_resources
import os
import glob
import subprocess
import time


def find_script_names(script_dir):
    script_paths = glob.glob(os.path.join(script_dir, "*.py"))

    script_filenames = [os.path.basename(s) for s in script_paths]
    _script_names = [os.path.splitext(s)[0] for s in script_filenames]
    script_names = []
    for sn in _script_names:
        if str.isdigit(sn[0:4]):
            script_names.append(sn)
    script_names.sort()
    return script_names


def find_job_dependencies(script_dir, script_names):
    job_dependencies = {}
    for script_name in script_names:
        job_dependencies[script_name] = []
        script_path = os.path.join(script_dir, script_name + ".py")
        with open(script_path, "rt") as fin:
            code = fin.read()
            for sn in script_names:
                if str.find(code, sn) >= 0:
                    job_dependencies[script_name].append(sn)
    return job_dependencies


def num_jobs(job_statii, status):
    num = 0
    for name in job_statii:
        if job_statii[name] == status:
            num += 1
    return num


def find_jobs_ready_to_run(job_statii, job_dependencies):
    jobs_ready_to_run = []
    for name in job_statii:
        if job_statii[name] == "pending":
            num_complete = 0
            for dep_name in job_dependencies[name]:
                if job_statii[dep_name] == "complete":
                    num_complete += 1
            if num_complete == len(job_dependencies[name]):
                jobs_ready_to_run.append(name)
    jobs_ready_to_run.sort()
    return jobs_ready_to_run


def run_parallel(run_dir, num_threads=6, polling_interval=1):
    script_dir = pkg_resources.resource_filename(
        "plenoirf", os.path.join("summary", "scripts")
    )

    script_names = find_script_names(script_dir=script_dir)
    job_dependencies = find_job_dependencies(
        script_dir=script_dir, script_names=script_names,
    )
    job_statii = {}
    job_handles = {}
    for name in script_names:
        job_statii[name] = "pending"
        job_handles[name] = None

    num_polls = 0
    while True:
        if num_jobs(job_statii, "error"):
            break

        if num_jobs(job_statii, "complete") == len(job_statii):
            break

        num_free_threads = num_threads - num_jobs(job_statii, "running")
        jobs_ready_to_run = find_jobs_ready_to_run(
            job_statii, job_dependencies
        )
        num_jobs_to_submit = min([len(jobs_ready_to_run), num_free_threads])

        for ii in range(num_jobs_to_submit):
            name = jobs_ready_to_run[ii]
            script_path = os.path.join(script_dir, name + ".py")
            pii = subprocess.Popen(args=["python", script_path, run_dir])
            job_statii[name] = "running"
            job_handles[name] = pii

        for name in script_names:
            job_popen = job_handles[name]
            if job_popen is not None:
                rc = job_popen.poll()

                if rc is None:
                    assert job_statii[name] == "running"
                else:
                    if rc >= 0:
                        job_statii[name] = "complete"
                        job_handles[name] = None
                    else:
                        job_statii[name] = "error"
                        job_handles[name] = None

        print("====================", num_polls)
        for name in script_names:
            print("{:<70s}     {:s}".format(name, job_statii[name]))

        time.sleep(polling_interval)
        num_polls += 1
