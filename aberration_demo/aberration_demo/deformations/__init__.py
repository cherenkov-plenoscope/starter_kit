"""
Explore how good a light-field-sensor can compensate deformations.

Alignment: perfect
Deformations: Yes, multiple magnitudes
Offaxisangles: few, only up to Portal's fov.
Mirror-geometries: Only parabola_segmented
"""
import os
import copy
import numpy as np
import json_numpy
import json_line_logger
import shutil
import plenoirf
import plenopy
import network_file_system as nfs

from . import parabola_segmented
from . import scenery
from .. import merlict
from .. import portal
from .. import analysis
from .. import calibration_source
from ..offaxis import read_config
from ..offaxis import PAXEL_FMT
from ..offaxis import ANGLE_FMT
from ..offaxis import guess_scaling_of_num_photons_used_to_estimate_light_field_geometry


CONFIG = {}
CONFIG["seed"] = 1337
CONFIG["executables"] = copy.deepcopy(merlict.EXECUTABLES)

CONFIG["mirror"] = copy.deepcopy(parabola_segmented.MIRROR)
CONFIG["sensor"] = copy.deepcopy(portal.SENSOR)
CONFIG["deformation_polynom"] = copy.deepcopy(
    parabola_segmented.DEFORMATION_POLYNOM
)

CONFIG["sources"] = {}
CONFIG["sources"]["off_axis_angles_deg"] = np.linspace(0.0, 3.0, 7)
CONFIG["sources"]["num_photons"] = 100 * 1000

CONFIG["light_field_geometry"] = {}
CONFIG["light_field_geometry"]["num_blocks"] = 1
CONFIG["light_field_geometry"]["num_photons_per_block"] = 100 * 1000

CONFIG["binning"] = copy.deepcopy(analysis.BINNING)


def init(work_dir, config=CONFIG):
    os.makedirs(work_dir, exist_ok=True)
    nfs.write(
        json_numpy.dumps(config, indent=4),
        os.path.join(work_dir, "config.json"),
        "wt",
    )
    nfs.write(
        json_numpy.dumps(merlict.PROPAGATION_CONFIG, indent=4),
        os.path.join(work_dir, "merlict_propagation_config.json"),
        "wt",
    )


def run(
    work_dir,
    map_and_reduce_pool,
    logger=json_line_logger.LoggerStdout(),
):
    """
    Runs the entire exploration.

    Parameters
    ----------
    work_dir : str
        The path to the work_dir.
    map_and_reduce_pool : pool
        Must have a map()-function. Used for parallel computing.
    logger : logging.Logger
        A logger of your choice.
    """
    logger.info("Start")

    logger.info("Make sceneries")
    make_sceneries_for_light_field_geometires(work_dir=work_dir)

    logger.info("Estimate light_field_geometry")
    make_light_field_geometires(
        work_dir=work_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        logger=logger,
    )

    logger.info("Make calibration source")
    make_source(work_dir=work_dir)

    logger.info("Make responses to calibration source")
    make_responses(work_dir=work_dir, map_and_reduce_pool=map_and_reduce_pool)

    logger.info("Make analysis")
    make_analysis(work_dir=work_dir, map_and_reduce_pool=map_and_reduce_pool)

    logger.info("Stop")


def make_sceneries_for_light_field_geometires(work_dir):
    config = read_config(work_dir=work_dir)

    geometries_dir = os.path.join(work_dir, "geometries")
    os.makedirs(geometries_dir, exist_ok=True)

    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        pkey = PAXEL_FMT.format(npax)
        pdir = os.path.join(geometries_dir, pkey)

        scenery_dir = os.path.join(pdir, "input", "scenery")
        os.makedirs(scenery_dir, exist_ok=True)
        with open(
            os.path.join(scenery_dir, "scenery.json"), "wt"
        ) as f:
            s = scenery.make_plenoscope_scenery_aligned_deformed(
                mirror_config=config["mirror"],
                deformation_polynom=config["deformation_polynom"],
                sensor_config=config["sensor"],
                num_paxel_on_diagonal=npax,
            )
            f.write(json_numpy.dumps(s, indent=4))


def make_light_field_geometires(
    work_dir, map_and_reduce_pool, logger,
):
    logger.info("lfg: Make jobs to estimate light-field-geometries.")
    jobs, rjobs = _light_field_geometries_make_jobs_and_rjobs(
        work_dir=work_dir
    )
    logger.info(
        "lfg: num jobs: mapping {:d}, reducing {:d}".format(
            len(jobs), len(rjobs)
        )
    )
    logger.info("lfg: Map")
    _ = map_and_reduce_pool.map(
        plenoirf.production.light_field_geometry.run_job, jobs
    )
    logger.info("lfg: Reduce")
    _ = map_and_reduce_pool.map(_light_field_geometries_run_rjob, rjobs)
    logger.info("lfg: Done")



def _light_field_geometries_make_jobs_and_rjobs(work_dir):
    config = read_config(work_dir=work_dir)

    jobs = []
    rjobs = []

    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        pkey = PAXEL_FMT.format(npax)
        pdir = os.path.join(work_dir, "geometries", pkey)

        out_dir = os.path.join(pdir, "light_field_geometry")

        if not os.path.exists(out_dir):

            # mapping
            # -------
            map_dir = os.path.join(pdir, "light_field_geometry.map")
            os.makedirs(map_dir, exist_ok=True)

            _num_blocks = config["light_field_geometry"]["num_blocks"]
            _num_blocks *= guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
                num_paxel_on_diagonal=npax
            )

            _num_photons_per_block = config["light_field_geometry"][
                "num_photons_per_block"
            ]

            _jobs = plenoirf.production.light_field_geometry.make_jobs(
                merlict_map_path=config["executables"][
                    "merlict_plenoscope_calibration_map_path"
                ],
                scenery_path=os.path.join(pdir, "input", "scenery"),
                map_dir=map_dir,
                num_photons_per_block=_num_photons_per_block,
                num_blocks=_num_blocks,
                random_seed=0,
            )

            jobs += _jobs

            # reducing
            # --------
            rjob = {}
            rjob["work_dir"] = work_dir
            rjob["pkey"] = pkey
            rjobs.append(rjob)

    return jobs, rjobs


def _light_field_geometries_run_rjob(rjob):
    config = read_config(work_dir=rjob["work_dir"])

    pdir = os.path.join(
        rjob["work_dir"],
        "geometries",
        rjob["pkey"],
    )

    map_dir = os.path.join(pdir, "light_field_geometry.map")
    out_dir = os.path.join(pdir, "light_field_geometry")

    rc = plenoirf.production.light_field_geometry.reduce(
        merlict_reduce_path=config["executables"][
            "merlict_plenoscope_calibration_reduce_path"
        ],
        map_dir=map_dir,
        out_dir=out_dir,
    )

    if rc == 0:
        shutil.rmtree(map_dir)

    return rc


def make_source(work_dir):
    """
    Makes the calibration-source.
    This is a bundle of parallel photons coming from zenith.
    It is written to work_dir/source.tar and is in the CORSIKA-like format
    EvnetTape.

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    """
    config = read_config(work_dir=work_dir)
    sources_dir = os.path.join(work_dir, "sources")
    os.makedirs(sources_dir, exist_ok=True)

    for iofa, off_axis_angle_deg in enumerate(config["sources"]["off_axis_angles_deg"]):
        source_path = os.path.join(sources_dir, ANGLE_FMT.format(iofa) + ".tar")

        if not os.path.exists(source_path):
            prng = np.random.Generator(np.random.PCG64(config["seed"]))
            off_axis_angle = np.deg2rad(off_axis_angle_deg)
            calibration_source.write_photon_bunches(
                cx=off_axis_angle,
                cy=0.0,
                size=config["sources"]["num_photons"],
                path=source_path,
                prng=prng,
                aperture_radius=1.2 * config["mirror"]["outer_radius"],
            )


def make_responses(
    work_dir, map_and_reduce_pool,
):
    """
    Makes the responses of the instruments to the calibration-sources.

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    map_and_reduce_pool : pool
        Used for parallel computing.
    """
    jobs = _responses_make_jobs(work_dir=work_dir)
    _ = map_and_reduce_pool.map(_responses_run_job, jobs)


def _responses_make_jobs(work_dir):
    jobs = []

    config = read_config(work_dir=work_dir)

    runningseed = int(config["seed"])

    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        pkey = PAXEL_FMT.format(npax)

        for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
            akey = ANGLE_FMT.format(ofa)

            job = {}
            job["work_dir"] = work_dir
            job["pkey"] = pkey
            job["akey"] = akey
            job["merlict_plenoscope_propagator_path"] = config[
                "executables"
            ]["merlict_plenoscope_propagator_path"]
            job["seed"] = runningseed
            jobs.append(job)

            runningseed += 1

    return jobs


def _responses_run_job(job):
    adir = os.path.join(
        job["work_dir"], "responses", job["pkey"], job["akey"]
    )
    os.makedirs(adir, exist_ok=True)
    response_event_path = os.path.join(adir, "1")

    if not os.path.exists(response_event_path):
        source_path = os.path.join(
            job["work_dir"], "sources", job["akey"] + ".tar"
        )
        plenoirf.production.merlict.plenoscope_propagator(
            corsika_run_path=source_path,
            output_path=adir,
            light_field_geometry_path=os.path.join(
                job["work_dir"],
                "geometries",
                job["pkey"],
                "light_field_geometry",
            ),
            merlict_plenoscope_propagator_path=job[
                "merlict_plenoscope_propagator_path"
            ],
            merlict_plenoscope_propagator_config_path=os.path.join(
                job["work_dir"], "merlict_propagation_config.json"
            ),
            random_seed=job["seed"],
            photon_origins=True,
            stdout_path=adir + ".o",
            stderr_path=adir + ".e",
        )

    left_over_input_dir = os.path.join(adir, "input")
    if os.path.exists(left_over_input_dir):
        shutil.rmtree(left_over_input_dir)

    return 1


def _analysis_make_jobs(
    work_dir, containment_percentile=80, object_distance_m=1e6,
):
    assert 0.0 < containment_percentile <= 100.0
    assert object_distance_m > 0.0

    config = read_config(work_dir=work_dir)

    jobs = []
    runningseed = int(config["seed"])

    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        pkey = PAXEL_FMT.format(npax)
        for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
            akey = ANGLE_FMT.format(ofa)
            angle_deg = config["sources"]["off_axis_angles_deg"][ofa]

            job = {}
            job["work_dir"] = work_dir
            job["pkey"] = pkey
            job["akey"] = akey
            job["off_axis_angle_deg"] = angle_deg
            job["seed"] = runningseed
            job["object_distance_m"] = object_distance_m
            job["containment_percentile"] = containment_percentile
            jobs.append(job)
            runningseed += 1

    return jobs


def _analysis_run_job(job):
    adir = os.path.join(
        job["work_dir"], "analysis", job["pkey"], job["akey"]
    )
    os.makedirs(adir, exist_ok=True)
    summary_path = os.path.join(adir, "summary.json")

    if os.path.exists(summary_path):
        return 1

    config = read_config(work_dir=job["work_dir"])
    prng = np.random.Generator(np.random.PCG64(job["seed"]))
    light_field_geometry = plenopy.LightFieldGeometry(
        path=os.path.join(
            job["work_dir"],
            "geometries",
            job["pkey"],
            "light_field_geometry",
        ),
    )
    event = plenopy.Event(
        path=os.path.join(
            job["work_dir"],
            "responses",
            job["pkey"],
            job["akey"],
            "1",
        ),
        light_field_geometry=light_field_geometry,
    )
    out = analysis.analyse_response_to_calibration_source(
        off_axis_angle_deg=job["off_axis_angle_deg"],
        event=event,
        light_field_geometry=light_field_geometry,
        object_distance_m=job["object_distance_m"],
        containment_percentile=job["containment_percentile"],
        binning=config["binning"],
        prng=prng,
    )
    nfs.write(json_numpy.dumps(out), summary_path, "wt")
    return 1


def make_analysis(
    work_dir,
    map_and_reduce_pool,
    object_distance_m=1e6,
    containment_percentile=80,
):
    jobs = _analysis_make_jobs(
        work_dir=work_dir,
        object_distance_m=object_distance_m,
        containment_percentile=containment_percentile,
    )
    _ = map_and_reduce_pool.map(_analysis_run_job, jobs)
