import json_line_logger
import os
import json_numpy
import plenopy
from . import observations
from ... import sources


def run(work_dir, pool, logger=json_line_logger.LoggerStdout()):
    logger.info("Analysis: Analyse responses")
    ajobs = _analysis_make_jobs(work_dir=work_dir)
    logger.info("Analysis: {:d} jobs to do".format(len(ajobs)))
    pool.map(_analysys_run_job, ajobs)
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
