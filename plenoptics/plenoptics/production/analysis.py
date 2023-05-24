import json_line_logger
import os
import glob
import json_numpy
import plenopy
from . import observations
from .. import sources


def run(work_dir, pool, logger=json_line_logger.LoggerStdout()):
    logger.info("Analysis: Analyse responses")
    ajobs = _analysis_make_jobs(work_dir=work_dir)
    logger.info("Analysis: {:d} jobs to do".format(len(ajobs)))
    pool.map(_analysys_run_job, ajobs)

    logger.info("Analysis: Reduce responses")
    rjobs = _analysis_reduce_make_jobs(work_dir=work_dir)
    logger.info("Analysis: {:d} jobs to do".format(len(rjobs)))
    pool.map(_analysis_reduce_run_job, rjobs)
    logger.info("Analysis:Done")


def _analysis_make_jobs(work_dir):
    return observations._tasks_make_jobs(
        work_dir=work_dir, task_key="analysis", suffix=".json"
    )


def _analysys_run_job(job):
    if job["observation_key"] == "star":
        sources.star.analysis_run_job(job=job)
    elif job["observation_key"] == "point":
        sources.point.analysis_run_job(job=job)
    elif job["observation_key"] == "phantom":
        pass
    else:
        raise AssertionError("Bad observation_key")


def _analysis_reduce_make_jobs(work_dir, task_key="analysis"):
    cfg_dir = os.path.join(work_dir, "config")
    config = json_numpy.read_tree(cfg_dir)
    jobs = []
    for instrument_key in config["observations"]["instruments"]:
        if instrument_key not in config["instruments"]:
            continue
        for observation_key in config["observations"]["instruments"][
            instrument_key
        ]:
            outpath = os.path.join(
                work_dir, task_key, instrument_key, observation_key + ".json",
            )
            if not os.path.exists(outpath):
                jobs.append(
                    {
                        "work_dir": work_dir,
                        "instrument_key": instrument_key,
                        "observation_key": observation_key,
                    }
                )
    return jobs


def _analysis_reduce_run_job(job):
    mapdir = os.path.join(
        job["work_dir"],
        "analysis",
        job["instrument_key"],
        job["observation_key"],
    )
    outpath = mapdir + ".json"
    return reduce_analysis_jobs(mapdir=mapdir, outpath=outpath)


def reduce_analysis_jobs(mapdir, outpath):
    paths = glob.glob(os.path.join(mapdir, "*.json"))
    paths.sort()

    out = {}
    for path in paths:
        basename = os.path.basename(path)
        number = str.split(basename, ".")[0]
        out[number] = json_numpy.read(path)

    json_numpy.write(outpath + ".incomplete", out, indent=None)
    os.rename(outpath + ".incomplete", outpath)
