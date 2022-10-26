"""
Explore how good a light-field-sensor can compensate deformations.

Alignment: perfect
Deformations: Yes, multiple magnitudes
Offaxisangles: few, only up to Portal's fov.
Mirror-geometries: Only parabola_segmented
"""
import copy
import json_numpy
import json_line_logger

from . import parabola_segmented
from . import scenery
from .. import merlict
from .. import portal
from ..offaxis import read_config
from ..offaxis import PAXEL_FMT


CONFIG = {}
CONFIG["seed"] = 1337
CONFIG["executables"] = copy.deepcopy(merlict.EXECUTABLES)
CONFIG["mirror"] = copy.deepcopy(parabola_segmented.CONFIG)
CONFIG["sensor"] = copy.deepcopy(portal.SENSOR)
CONFIG["deformation_polynom"] = copy.deepcopy(
    parabola_segmented.DEFORMATION_POLYNOM
)


def init(work_dir, config=CONFIG):
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "config.json"), "wt") as f:
        f.write(json_numpy.dumps(config, indent=4))
    with open(
        os.path.join(work_dir, "merlict_propagation_config.json"), "wt"
    ) as f:
        f.write(json_numpy.dumps(merlict.PROPAGATION_CONFIG, indent=4))


def run(
    work_dir,
    map_and_reduce_pool,
    logger=json_line_logger.LoggerStdout(),
    desired_num_bunbles=400,
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

    logger.info("Stop")


def make_sceneries_for_light_field_geometires(work_dir):
    config = read_config(work_dir=work_dir)

    geometries_dir = os.path.join(work_dir, "geometries")
    os.makedirs(geometries_dir, exist_ok=True)

    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        pkey = PAXEL_FMT.format(npax)
        pdir = os.path.join(mdir, pkey)

        scenery_dir = os.path.join(pdir, "input", "scenery")
        os.makedirs(scenery_dir, exist_ok=True)
        with open(
            os.path.join(scenery_dir, "scenery.json"), "wt"
        ) as f:
            s = scenery.make_plenoscope_scenery_aligned_deformed(
                mirror_config=config["mirror"],
                deformation_polygon=deformation_polygon,
                sensor_config=config["sensor"],
                num_paxel_on_diagonal=npax,
            )
            f.write(json_numpy.dumps(s, indent=4))
