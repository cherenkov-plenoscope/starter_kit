import os
import json_numpy
import copy
import numpy as np
import json_line_logger
import plenoirf
import plenopy
import shutil
from ... import deformations
from ... import portal
from ... import merlict
from ... import utils


def run(work_dir, pool, logger=json_line_logger.LoggerStdout()):
    logger.info("lfg: Start")
    logger.info("lfg: Make sceneries")
    sjobs = make_sceneries_make_jobs(work_dir=work_dir)
    logger.info("lfg: {:d} jobs to do".format(len(sjobs)))
    pool.map(make_sceneries_run_job, sjobs)
    logger.info("lfg: Sceneries done")

    logger.info("lfg: Populating statistics of beams")
    mjobs, rjobs = map_and_reduce_make_jobs(work_dir=work_dir)
    logger.info("lfg: {:d} jobs to do".format(len(mjobs)))
    pool.map(plenoirf.production.light_field_geometry.run_job, mjobs)
    pool.map(reduce_run_job, rjobs)
    logger.info("lfg: Statistics of beams done")

    logger.info("lfg: Make plots")
    pjobs = plot_make_jobs(work_dir=work_dir)
    logger.info("lfg: {:d} jobs to do".format(len(pjobs)))
    pool.map(plot_run_job, pjobs)
    logger.info("lfg: Plots Done")
    logger.info("lfg: Done")


def make_sceneries_make_jobs(work_dir):
    config = json_numpy.read_tree(os.path.join(work_dir, "config"))
    instruments_dir = os.path.join(work_dir, "instruments")
    jobs = []
    for instrument_key in config["instruments"]:
        instrument_dir = os.path.join(instruments_dir, instrument_key)

        if not os.path.exists(instrument_dir):
            job = {
                "work_dir": work_dir,
                "instrument_key": instrument_key,
            }
            jobs.append(job)
    return jobs


def make_sceneries_run_job(job):
    config = json_numpy.read_tree(os.path.join(job["work_dir"], "config"))
    instrument_dir = os.path.join(
        job["work_dir"], "instruments", job["instrument_key"]
    )
    os.makedirs(instrument_dir, exist_ok=True)

    ikey = job["instrument_key"]
    icfg = config["instruments"][ikey]

    mirror_dimensions = config["mirrors"][icfg["mirror"]]
    sensor_dimensions = config["sensors"][icfg["sensor"]]
    mirror_deformation = config["mirror_deformations"][
        icfg["mirror_deformation"]
    ]
    sensor_transformation = config["sensor_transformations"][
        icfg["sensor_transformation"]
    ]
    num_paxel_on_pixel_diagonal = sensor_dimensions[
        "num_paxel_on_pixel_diagonal"
    ]

    mirror_deformation_map = deformations.deformation_map.init_from_mirror_and_deformation_configs(
        mirror_dimensions=mirror_dimensions,
        mirror_deformation=mirror_deformation,
    )

    merlict_scenery = deformations.scenery.make_plenoscope_scenery_aligned_deformed(
        mirror_dimensions=mirror_dimensions,
        mirror_deformation_map=mirror_deformation_map,
        sensor_dimensions=sensor_dimensions,
        sensor_transformation=sensor_transformation,
        num_paxel_on_pixel_diagonal=num_paxel_on_pixel_diagonal,
    )

    scenery_dir = os.path.join(instrument_dir, "input", "scenery")
    os.makedirs(scenery_dir, exist_ok=True)
    json_numpy.write(
        os.path.join(scenery_dir, "scenery.json"), merlict_scenery
    )


def map_and_reduce_make_jobs(work_dir):
    config = json_numpy.read_tree(os.path.join(work_dir, "config"))
    instruments_dir = os.path.join(work_dir, "instruments")

    jobs = []
    rjobs = []

    for instrument_key in config["instruments"]:
        instrument_dir = os.path.join(instruments_dir, instrument_key)
        light_field_geometry_dir = os.path.join(
            instrument_dir, "light_field_geometry"
        )

        if not os.path.exists(light_field_geometry_dir):

            map_dir = os.path.join(instrument_dir, "light_field_geometry.map")
            os.makedirs(map_dir, exist_ok=True)

            icfg = config["instruments"][instrument_key]
            sensor_dimensions = config["sensors"][icfg["sensor"]]
            num_paxel_on_pixel_diagonal = sensor_dimensions[
                "num_paxel_on_pixel_diagonal"
            ]
            _num_blocks = config["statistics"]["light_field_geometry"][
                "num_blocks"
            ]
            _num_blocks *= utils.guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
                num_paxel_on_pixel_diagonal=num_paxel_on_pixel_diagonal
            )

            _jobs = plenoirf.production.light_field_geometry.make_jobs(
                merlict_map_path=config["merlict"]["executables"][
                    "merlict_plenoscope_calibration_map_path"
                ],
                scenery_path=os.path.join(instrument_dir, "input", "scenery"),
                map_dir=map_dir,
                num_photons_per_block=config["statistics"][
                    "light_field_geometry"
                ]["num_photons_per_block"],
                num_blocks=_num_blocks,
                random_seed=0,
            )
            jobs += _jobs

            # reducing
            # --------
            rjob = {}
            rjob["work_dir"] = work_dir
            rjob["instrument_key"] = instrument_key
            rjobs.append(rjob)

    return jobs, rjobs


def reduce_run_job(job):
    config = json_numpy.read_tree(os.path.join(job["work_dir"], "config"))

    instrument_dir = os.path.join(
        job["work_dir"], "instruments", job["instrument_key"],
    )

    map_dir = os.path.join(instrument_dir, "light_field_geometry.map")
    out_dir = os.path.join(instrument_dir, "light_field_geometry")

    rc = plenoirf.production.light_field_geometry.reduce(
        merlict_reduce_path=config["merlict"]["executables"][
            "merlict_plenoscope_calibration_reduce_path"
        ],
        map_dir=map_dir,
        out_dir=out_dir,
    )

    if rc == 0:
        shutil.rmtree(map_dir)

    input_dir = os.path.join(instrument_dir, "input")
    shutil.rmtree(input_dir)

    return rc


def plot_make_jobs(work_dir):
    config = json_numpy.read_tree(os.path.join(work_dir, "config"))
    instruments_dir = os.path.join(work_dir, "instruments")

    jobs = []

    for instrument_key in config["instruments"]:
        instrument_dir = os.path.join(instruments_dir, instrument_key)
        lfg_dir = os.path.join(instrument_dir, "light_field_geometry")
        plot_dir = os.path.join(lfg_dir, "plot")

        if not os.path.exists(plot_dir):
            job = {}
            job["work_dir"] = work_dir
            job["instrument_key"] = instrument_key
            jobs.append(job)

    return jobs


def plot_run_job(job):
    lfg_dir = os.path.join(
        job["work_dir"],
        "instruments",
        job["instrument_key"],
        "light_field_geometry",
    )
    plot_dir = os.path.join(lfg_dir, "plot")
    try:
        lfg = plenopy.light_field_geometry.LightFieldGeometry(path=lfg_dir)
        plenopy.plot.light_field_geometry.save_all(
            light_field_geometry=lfg, out_dir=plot_dir,
        )
    except Exception as e:
        print(e)
