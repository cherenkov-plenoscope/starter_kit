from . import default_config
from . import production
from . import sources
from . import analysis

import os
import numpy as np
import json_numpy
import json_line_logger
import plenopy
import pkg_resources
import subprocess


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

    logger.info("Plot mirror deformations")
    pjobs = _plot_mirror_deformations_make_jobs(work_dir=work_dir)
    logger.info("{:d} jobs to do".format(len(pjobs)))
    pool.map(_run_script_job, pjobs)

    logger.info("Plot guide stars")
    plot_guide_stars(work_dir=work_dir, pool=pool, logger=logger)

    _run_script(
        script="plot_image_of_star_vs_offaxis",
        argv=[
            "--work_dir",
            work_dir,
            "--out_dir",
            os.path.join(work_dir, "plots", "guide_stars_vs_offaxis"),
        ],
    )

    logger.info("Plot depth")
    pjobs = _plot_depth_make_jobs(work_dir=work_dir)
    logger.info("{:d} jobs to do".format(len(pjobs)))
    pool.map(_run_script_job, pjobs)

    logger.info("Plot phantom")
    pjobs = _plot_phantom_source_make_jobs(work_dir=work_dir)
    logger.info("{:d} jobs to do".format(len(pjobs)))
    pool.map(_run_script_job, pjobs)

    logger.info("Plots done")
    logger.info("Done")


def _plot_mirror_deformations_make_jobs(work_dir):
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


def _plot_depth_make_jobs(work_dir):
    cfg_dir = os.path.join(work_dir, "config")
    config = json_numpy.read_tree(cfg_dir)

    jobs = []
    for instrument_key in config["observations"]["instruments"]:
        if "point" in config["observations"]["instruments"][instrument_key]:
            depth_out_dir = os.path.join(
                work_dir, "plots", "depth", instrument_key
            )
            if not os.path.exists(depth_out_dir):
                job = {
                    "script": "plot_depth",
                    "argv": [
                        "--work_dir",
                        work_dir,
                        "--out_dir",
                        os.path.join(
                            work_dir, "plots", "depth", instrument_key
                        ),
                        "--instrument_key",
                        instrument_key,
                    ],
                }
                jobs.append(job)

            depth_refocus_out_dir = os.path.join(
                work_dir, "plots", "depth_refocus", instrument_key
            )
            if not os.path.exists(depth_refocus_out_dir):
                job = {
                    "script": "plot_depth_refocus",
                    "argv": [
                        "--work_dir",
                        work_dir,
                        "--out_dir",
                        depth_refocus_out_dir,
                        "--instrument_key",
                        instrument_key,
                    ],
                }
                jobs.append(job)
    return jobs


def _plot_phantom_source_make_jobs(work_dir):
    cfg_dir = os.path.join(work_dir, "config")
    config = json_numpy.read_tree(cfg_dir)

    jobs = []
    for instrument_key in config["observations"]["instruments"]:
        if "phantom" in config["observations"]["instruments"][instrument_key]:
            out_dir = os.path.join(
                work_dir, "plots", "phantom", instrument_key
            )

            if not os.path.exists(out_dir):
                job = {
                    "script": "plot_phantom_source",
                    "argv": [
                        "--work_dir",
                        work_dir,
                        "--out_dir",
                        out_dir,
                        "--instrument_key",
                        instrument_key,
                    ],
                }
                jobs.append(job)
    return jobs


def _run_script_job(job):
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


def plot_guide_stars(work_dir, pool, logger):
    out_dir = os.path.join(work_dir, "plots", "guide_stars")

    _run_script(
        script="plot_image_of_star_cmap",
        argv=["--work_dir", work_dir, "--out_dir", out_dir],
    )

    table_vmax = analysis.guide_stars.table_vmax(work_dir=work_dir)
    vmax = analysis.guide_stars.table_vmax_max(table_vmax=table_vmax)

    jobs = []
    for instrument_key in table_vmax:
        for star_key in table_vmax[instrument_key]:
            job = {"script": "plot_image_of_star"}
            job["argv"] = [
                "--work_dir",
                work_dir,
                "--out_dir",
                out_dir,
                "--instrument_key",
                instrument_key,
                "--star_key",
                star_key,
                "--vmax",
                "{:e}".format(vmax),
            ]
            jobs.append(job)

    pool.map(_run_script_job, jobs)
