from . import parabola_segmented
from . import scenery
from .. import merlict
from .. import portal


CONFIG = {}
CONFIG["seed"] = 1337
CONFIG["executables"] = dict(merlict.EXECUTABLES)
CONFIG["mirror"] = dict(portal.MIRROR)
CONFIG["sensor"] = dict(portal.SENSOR)


def init(work_dir, config):
    pass