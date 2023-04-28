import os
import numpy as np
import json_numpy
import json_line_logger
import corsika_primary
import plenopy

from . import default_config
from . import production
from .. import sources


def init(work_dir, random_seed=42, minimal=False):
    cfg_dir = os.path.join(work_dir, "config")
    default_config.write_default_config(cfg_dir=cfg_dir, minimal=minimal)


def run(work_dir, pool, logger=json_line_logger.LoggerStdout()):
    logger.info("Start")

    logger.info("Make light-field-geometryies")
    production.light_field_geometry.run(
        work_dir=work_dir, pool=pool, logger=logger
    )
    logger.info("Light-field-geometryies done")

    logger.info("Make observations")
    production.observations.run(work_dir=work_dir, pool=pool, logger=logger)
    logger.info("Observations done")

    ajobs = _analysis_make_jobs(work_dir=work_dir)
    pool.map(_analysys_run_job, ajobs)

    logger.info("Done")


def _analysis_make_jobs(work_dir):
    return production.observations._tasks_make_jobs(
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
