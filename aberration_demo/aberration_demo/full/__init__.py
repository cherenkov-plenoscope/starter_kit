import os
import json_line_logger

from . import default_config
from . import production


def init(work_dir, random_seed=42, minimal=False):
    cfg_dir = os.path.join(work_dir, "config")
    default_config.write_default_config(cfg_dir=cfg_dir, minimal=minimal)


def run(work_dir, pool, logger=json_line_logger.LoggerStdout()):
    production.light_field_geometry.run(
        work_dir=work_dir, pool=pool, logger=logger
    )
