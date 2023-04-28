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

    logger.info("Make Analysis")
    production.analysis.run(work_dir=work_dir, pool=pool, logger=logger)
    logger.info("Analysis done")

    logger.info("Done")
