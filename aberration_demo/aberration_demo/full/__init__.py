import os
import numpy as np
import json_numpy
import json_line_logger
import corsika_primary
import plenopy
import pkg_resources
import subprocess

from . import default_config
from . import production
from . import plots
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

    logger.info("Make Analysis")
    production.analysis.run(work_dir=work_dir, pool=pool, logger=logger)
    logger.info("Analysis done")

    logger.info("Make Plots")
    pjobs = _plot_make_jobs(work_dir=work_dir)
    logger.info("{:d} jobs to do".format(len(pjobs)))
    pool.map(_plot_run_job, pjobs)
    logger.info("Plots done")

    logger.info("Done")


def _plot_make_jobs(work_dir):
    cfg_dir = os.path.join(work_dir, "config")
    config = json_numpy.read_tree(cfg_dir)

    jobs = []

    # mirrors
    # -------
    for mirror_key in config["mirrors"]:
        mirror_dimensions_path = os.path.join(
            work_dir, "config", "mirrors", mirror_key + ".json"
        )
        for deformation_key in config["mirror_deformations"]:

            outpath = os.path.join(
                work_dir, "plots", "mirrors", mirror_key, deformation_key,
            )

            mirror_deformations_path = os.path.join(
                work_dir,
                "config",
                "mirror_deformations",
                deformation_key + ".json",
            )

            if not os.path.exists(outpath):
                job = {
                    "script": "plot_mirror_deformation",
                    "argv": [
                        mirror_dimensions_path,
                        mirror_deformations_path,
                        outpath,
                    ],
                }
                jobs.append(job)

    return jobs


def _plot_run_job(job):
    return _run_script(script=job["script"], argv=job["argv"])


def _run_script(script, argv):
    if not script.endswith(".py"):
        script += ".py"

    script_path = pkg_resources.resource_filename(
        "aberration_demo", os.path.join("full", "scripts", script),
    )
    args = []
    args.append("python")
    args.append(script_path)
    args += argv
    return subprocess.call(args)
