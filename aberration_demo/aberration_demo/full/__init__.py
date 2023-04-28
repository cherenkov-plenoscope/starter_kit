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
    production.light_field_geometry.run(
        work_dir=work_dir, pool=pool, logger=logger
    )

    ojobs = _observations_make_jobs(work_dir=work_dir)
    pool.map(_observations_run_job, ojobs)


def _observations_make_jobs(work_dir):
    cfg_dir = os.path.join(work_dir, "config")
    confg = json_numpy.read_tree(cfg_dir)

    jobs = []

    for instrument_key in confg["observations"]["instruments"]:

        if instrument_key not in confg["instruments"]:
            continue

        for observation_key in confg["observations"]["instruments"][
            instrument_key
        ]:
            if observation_key == "star":
                stars = confg["observations"]["star"]
                for n in range(stars["num_stars"]):
                    nkey = "{:06d}".format(n)
                    outpath = os.path.join(
                        work_dir,
                        "responses",
                        instrument_key,
                        observation_key,
                        nkey,
                    )
                    if not os.path.exists(outpath):
                        job = {
                            "work_dir": work_dir,
                            "instrument_key": instrument_key,
                            "observation_key": observation_key,
                            "number": n,
                        }
                        jobs.append(job)

            elif observation_key == "point":
                points = confg["observations"]["point"]
                for n in range(points["num_points"]):
                    nkey = "{:06d}".format(n)
                    outpath = os.path.join(
                        work_dir,
                        "responses",
                        instrument_key,
                        observation_key,
                        nkey,
                    )
                    if not os.path.exists(outpath):
                        job = {
                            "work_dir": work_dir,
                            "instrument_key": instrument_key,
                            "observation_key": observation_key,
                            "number": n,
                        }
                        jobs.append(job)
            elif observation_key == "phantom":
                phantom = confg["observations"]["phantom"]
                n = 0

                outpath = os.path.join(
                    work_dir,
                    "responses",
                    instrument_key,
                    observation_key,
                    nkey,
                )
                if not os.path.exists(outpath):
                    job = {
                        "work_dir": work_dir,
                        "instrument_key": instrument_key,
                        "observation_key": observation_key,
                        "number": n,
                    }
                    jobs.append(job)
    return jobs


def _observations_run_job(job):
    outdir = os.path.join(
        job["work_dir"],
        "responses",
        job["instrument_key"],
        job["observation_key"],
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "{:06d}".format(job["number"]))
    light_field_geometry_path = os.path.join(
        job["work_dir"],
        "instruments",
        job["instrument_key"],
        "light_field_geometry",
    )

    merlict_config = json_numpy.read_tree(
        os.path.join(job["work_dir"], "config", "merlict")
    )

    if job["observation_key"] == "star":
        source_config = sources.star.make_source_config_from_job(job=job)
    elif job["observation_key"] == "point":
        source_config = sources.point.make_source_config_from_job(job=job)
    elif job["observation_key"] == "phantom":
        source_config = sources.mesh.make_source_config_from_job(job=job)
    else:
        raise AssertionError("Bad observation_key")

    raw_sensor_response = production.observations.make_response_to_source(
        source_config=source_config,
        light_field_geometry_path=light_field_geometry_path,
        merlict_config=merlict_config,
    )

    # export truth
    # ------------
    outtruthpath = outpath + ".json"
    json_numpy.write(
        outtruthpath + ".incomplete", source_config,
    )
    os.rename(outtruthpath + ".incomplete", outtruthpath)

    # export raw sensor resposnse
    # ---------------------------
    with open(outpath + ".incomplete", "wb") as f:
        plenopy.raw_light_field_sensor_response.write(
            f=f, raw_sensor_response=raw_sensor_response
        )
    os.rename(outpath + ".incomplete", outpath)
