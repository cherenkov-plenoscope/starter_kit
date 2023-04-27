"""
Explore how good a light-field-sensor can compensate aberrations.

Alignment: perfect
Deformations: None
Offaxisangles: Multiple and large
Mirror-geometries: Multiple
"""
import numpy as np
import plenoirf
import multiprocessing
import json_numpy
import os
import shutil
import plenopy
import json_line_logger
import copy
import network_file_system as nfs

from . import scenery
from .. import merlict
from .. import calibration_source
from .. import portal
from .. import analysis
from .. import utils
from ..utils import read_json
from ..utils import read_config
from ..utils import PAXEL_FMT
from ..utils import ANGLE_FMT


CONFIG = {}
CONFIG["seed"] = 42

CONFIG["mirror"] = {}
CONFIG["mirror"]["dimensions"] = copy.deepcopy(portal.MIRROR)
CONFIG["mirror"]["keys"] = [
    "sphere_monolith",
    # "davies_cotton",
    "parabola_segmented",
]

CONFIG["sensor"] = {}
CONFIG["sensor"]["dimensions"] = copy.deepcopy(portal.SENSOR)
CONFIG["sensor"]["num_paxel_on_pixel_diagonal"] = [1, 3]

CONFIG["sources"] = {}
CONFIG["sources"]["off_axis_angles_deg"] = np.linspace(0.0, 8.0, 3)
CONFIG["sources"]["num_photons"] = 1000 * 1000

CONFIG["light_field_geometry"] = {}
CONFIG["light_field_geometry"]["num_blocks"] = 1
CONFIG["light_field_geometry"]["num_photons_per_block"] = 1000 * 1000
CONFIG["binning"] = copy.deepcopy(analysis.BINNING)


def make_config_from_scenery(scenery_path, seed=1337):
    scenery = read_json(scenery_path)
    (
        mirror_dimensions,
        sensor_dimensions,
    ) = merlict.make_mirror_and_sensor_dimensions_from_merlict_scenery(scenery)

    cfg = {}
    cfg["seed"] = seed

    cfg["mirror"] = {}
    cfg["mirror"]["dimensions"] = mirror_dimensions
    cfg["mirror"]["keys"] = CONFIG["mirror"]["keys"]

    cfg["sensor"] = {}
    cfg["sensor"]["dimensions"] = sensor_dimensions
    cfg["sensor"]["num_paxel_on_pixel_diagonal"] = CONFIG["sensor"][
        "num_paxel_on_pixel_diagonal"
    ]

    cfg["sources"] = CONFIG["sources"]
    cfg["light_field_geometry"] = CONFIG["light_field_geometry"]

    cfg["binning"] = CONFIG["binning"]
    return cfg


def init(work_dir, config=CONFIG, executables=merlict.EXECUTABLES):
    """
    Initialize the work_dir, i.e. the base of all operations to explore how
    plenoptics can compensate distortions and aberrations.

    When initialized, you might want to adjust the work_dir/config.json.
    Then you call run(work_dir) to run the exploration.

    Parameters
    ----------
    work_dir : str
        The path to the work_dir.
    config : dict
        The configuration of our explorations.
        Configure the different geometries of the mirrors,
        the different off-axis-angles,
        and the number of photo-sensors in the light-field-sensor.
    """
    os.makedirs(work_dir, exist_ok=True)

    nfs.write(
        json_numpy.dumps(executables, indent=4),
        os.path.join(work_dir, "executables.json"),
        "wt",
    )
    nfs.write(
        json_numpy.dumps(merlict.PROPAGATION_CONFIG, indent=4),
        os.path.join(work_dir, "merlict_propagation_config.json"),
        "wt",
    )
    nfs.write(
        json_numpy.dumps(config, indent=4),
        os.path.join(work_dir, "config.json"),
        "wt",
    )


def run(
    work_dir, map_and_reduce_pool, logger=json_line_logger.LoggerStdout(),
):
    """
    Runs the entire exploration.
        - Makes the sceneries with the optics
        - Estimates the light-field-geometries of the optical instruments.
        - Makes the calibration-source.
        - Makes the response of each instrument to the calibration-source.
        - Analyses each response.
        - Makes overview plots of the responses.

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


def LightFieldGeometry(path, off_axis_angle_deg):
    """
    Returns a plenopy.LightFieldGeometry(path) but de-rotated by
    off_axis_angle_deg in cx.

    Parameters
    ----------
    path : str
        Path to the plenopy.LightFieldGeometry
    off_axis_angle_deg : float
        The off-axis-angle of the light.
    """
    lfg = plenopy.LightFieldGeometry(path=path)
    lfg.cx_mean += np.deg2rad(off_axis_angle_deg)
    return lfg


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

    config = json_numpy.read(os.path.join(work_dir, "config.json"))
    executables = json_numpy.read(os.path.join(work_dir, "executables.json"))

    runningseed = int(config["seed"])
    for mkey in config["mirror"]["keys"]:

        for npax in config["sensor"]["num_paxel_on_pixel_diagonal"]:
            pkey = PAXEL_FMT.format(npax)

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)

                job = {}
                job["work_dir"] = work_dir
                job["mkey"] = mkey
                job["pkey"] = pkey
                job["akey"] = akey
                job["merlict_plenoscope_propagator_path"] = executables[
                    "merlict_plenoscope_propagator_path"
                ]
                job["seed"] = runningseed
                jobs.append(job)

                runningseed += 1

    return jobs


def _responses_run_job(job):
    adir = os.path.join(
        job["work_dir"], "responses", job["mkey"], job["pkey"], job["akey"]
    )
    os.makedirs(adir, exist_ok=True)
    response_event_path = os.path.join(adir, "1")

    if not os.path.exists(response_event_path):
        plenoirf.production.merlict.plenoscope_propagator(
            corsika_run_path=os.path.join(job["work_dir"], "source.tar"),
            output_path=adir,
            light_field_geometry_path=os.path.join(
                job["work_dir"],
                "geometries",
                job["mkey"],
                job["pkey"],
                job["akey"],
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

    config = json_numpy.read(os.path.join(work_dir, "config.json"))

    jobs = []
    runningseed = int(config["seed"])

    for mkey in config["mirror"]["keys"]:
        for npax in config["sensor"]["num_paxel_on_pixel_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)
                angle_deg = config["sources"]["off_axis_angles_deg"][ofa]

                job = {}
                job["work_dir"] = work_dir
                job["mkey"] = mkey
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
        job["work_dir"], "analysis", job["mkey"], job["pkey"], job["akey"]
    )
    os.makedirs(adir, exist_ok=True)
    summary_path = os.path.join(adir, "summary.json")

    if os.path.exists(summary_path):
        return 1

    config = json_numpy.read(os.path.join(job["work_dir"], "config.json"))
    prng = np.random.Generator(np.random.PCG64(job["seed"]))

    light_field_geometry = LightFieldGeometry(
        path=os.path.join(
            job["work_dir"],
            "geometries",
            job["mkey"],
            job["pkey"],
            job["akey"],
            "light_field_geometry",
        ),
        off_axis_angle_deg=job["off_axis_angle_deg"],
    )
    event = plenopy.Event(
        path=os.path.join(
            job["work_dir"],
            "responses",
            job["mkey"],
            job["pkey"],
            job["akey"],
            "1",
        ),
        light_field_geometry=light_field_geometry,
    )
    out = analysis.analyse_response_to_calibration_source(
        off_axis_angle_deg=job["off_axis_angle_deg"],
        raw_sensor_response=event.raw_sensor_response,
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
    config = json_numpy.read(os.path.join(work_dir, "config.json"))

    source_path = os.path.join(work_dir, "source.tar")

    if not os.path.exists(source_path):
        prng = np.random.Generator(np.random.PCG64(config["seed"]))
        calibration_source.write_photon_bunches(
            cx=0.0,
            cy=0.0,
            size=config["sources"]["num_photons"],
            path=source_path,
            prng=prng,
            aperture_radius=1.2
            * config["mirror"]["dimensions"]["max_outer_aperture_radius"],
        )


def make_sceneries_for_light_field_geometires(work_dir):
    """
    Makes the sceneries of the instruments for each combination of:
    - mirror-geometry
    - photo-sensor-density
    - off-axis-angle

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    """
    config = read_json(os.path.join(work_dir, "config.json"))

    geometries_dir = os.path.join(work_dir, "geometries")
    os.makedirs(geometries_dir, exist_ok=True)

    for mkey in config["mirror"]["keys"]:
        mdir = os.path.join(geometries_dir, mkey)

        for npax in config["sensor"]["num_paxel_on_pixel_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            pdir = os.path.join(mdir, pkey)

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)
                adir = os.path.join(pdir, akey)

                scenery_dir = os.path.join(adir, "input", "scenery")
                os.makedirs(scenery_dir, exist_ok=True)
                with open(
                    os.path.join(scenery_dir, "scenery.json"), "wt"
                ) as f:
                    s = scenery.make_plenoscope_scenery_for_merlict(
                        mirror_key=mkey,
                        num_paxel_on_pixel_diagonal=npax,
                        config=config,
                        off_axis_angles_deg=config["sources"][
                            "off_axis_angles_deg"
                        ][ofa],
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
    config = json_numpy.read(os.path.join(work_dir, "config.json"))
    executables = json_numpy.read(os.path.join(work_dir, "executables.json"))

    jobs = []
    rjobs = []

    for mkey in config["mirror"]["keys"]:
        for npax in config["sensor"]["num_paxel_on_pixel_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)

                adir = os.path.join(work_dir, "geometries", mkey, pkey, akey,)

                out_dir = os.path.join(adir, "light_field_geometry")

                if not os.path.exists(out_dir):

                    # mapping
                    # -------
                    map_dir = os.path.join(adir, "light_field_geometry.map")
                    os.makedirs(map_dir, exist_ok=True)

                    _num_blocks = config["light_field_geometry"]["num_blocks"]
                    _num_blocks *= utils.guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
                        num_paxel_on_pixel_diagonal=npax
                    )

                    _num_photons_per_block = config["light_field_geometry"][
                        "num_photons_per_block"
                    ]

                    _jobs = plenoirf.production.light_field_geometry.make_jobs(
                        merlict_map_path=executables[
                            "merlict_plenoscope_calibration_map_path"
                        ],
                        scenery_path=os.path.join(adir, "input", "scenery"),
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
                    rjob["mkey"] = mkey
                    rjob["pkey"] = pkey
                    rjob["akey"] = akey
                    rjobs.append(rjob)

    return jobs, rjobs


def _light_field_geometries_run_rjob(rjob):
    config = json_numpy.read(os.path.join(rjob["work_dir"], "config.json"))
    executables = json_numpy.read(
        os.path.join(rjob["work_dir"], "executables.json")
    )

    adir = os.path.join(
        rjob["work_dir"],
        "geometries",
        rjob["mkey"],
        rjob["pkey"],
        rjob["akey"],
    )

    map_dir = os.path.join(adir, "light_field_geometry.map")
    out_dir = os.path.join(adir, "light_field_geometry")

    rc = plenoirf.production.light_field_geometry.reduce(
        merlict_reduce_path=executables[
            "merlict_plenoscope_calibration_reduce_path"
        ],
        map_dir=map_dir,
        out_dir=out_dir,
    )

    if rc == 0:
        shutil.rmtree(map_dir)

    return rc


def read_analysis(work_dir):
    config = read_config(work_dir=work_dir)

    coll = {}
    for mkey in config["mirror"]["keys"]:
        coll[mkey] = {}

        for npax in config["sensor"]["num_paxel_on_pixel_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            coll[mkey][pkey] = {}

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)

                summary_path = os.path.join(
                    work_dir, "analysis", mkey, pkey, akey, "summary.json",
                )
                if not os.path.exists(summary_path):
                    print("Expected summary:", summary_path)
                    continue

                with open(summary_path, "rt") as f:
                    out = json_numpy.loads(f.read())
                coll[mkey][pkey][akey] = out
    return coll
