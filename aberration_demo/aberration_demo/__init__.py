from . import full
from . import offaxis
from . import deformations
from . import merlict
from . import sources
from . import portal
from . import analysis
from . import utils

import os


def init(work_dir):
    os.makedirs(work_dir, exist_ok=True)
    offaxis.init(work_dir=os.path.join(work_dir, "offaxis"))
    deformations.init(work_dir=os.path.join(work_dir, "deformations"))


def run(work_dir, map_and_reduce_pool, logger):
    offaxis.run(
        work_dir=os.path.join(work_dir, "offaxis"),
        map_and_reduce_pool=map_and_reduce_pool,
        logger=logger,
    )
    deformations.run(
        work_dir=os.path.join(work_dir, "deformations"),
        map_and_reduce_pool=map_and_reduce_pool,
        logger=logger,
    )
