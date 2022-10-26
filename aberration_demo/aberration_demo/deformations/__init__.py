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
import network_file_system as nfs

from . import parabola_segmented
from . import scenery
from .. import merlict
from .. import portal
from .. import analysis
from .. import calibration_source
from ..offaxis import read_config
from ..offaxis import PAXEL_FMT
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
CONFIG["sources"]["off_axis_angles_deg"] = np.linspace(0.0, 8.0, 9)
CONFIG["sources"]["num_photons"] = 1000 * 1000

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
        source_path = os.path.join(sources_dir, "{:06d}.tar".format(iofa))

        if not os.path.exists(source_path):
            prng = np.random.Generator(np.random.PCG64(config["seed"]))
            off_axis_angle = np.rad2deg(off_axis_angle_deg)
            calibration_source.write_photon_bunches(
                cx=off_axis_angle,
                cy=0.0,
                size=config["sources"]["num_photons"],
                path=source_path,
                prng=prng,
                aperture_radius=1.2 * config["mirror"]["outer_radius"],
            )
