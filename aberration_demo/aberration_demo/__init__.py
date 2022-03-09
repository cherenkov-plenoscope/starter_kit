"""
simulate different light-field-sensors
"""
import numpy as np
import plenoirf
import queue_map_reduce
import multiprocessing
import json_numpy
import os
from . import merlict


EXECUTABLES = {
    "merlict_plenoscope_propagator_path": os.path.abspath(os.path.join(
        "build", "merlict", "merlict-plenoscope-propagation"
    )),
    "merlict_plenoscope_calibration_map_path": os.path.abspath(os.path.join(
        "build", "merlict", "merlict-plenoscope-calibration-map"
    )),
    "merlict_plenoscope_calibration_reduce_path": os.path.abspath(os.path.join(
        "build", "merlict", "merlict-plenoscope-calibration-reduce"
    )),
}

CFG = {}
CFG["mirror"] = {}
CFG["mirror"]["focal_length"] = 106.5
CFG["mirror"]["inner_radius"] = 35.5
CFG["mirror"]["outer_radius"] = (2/np.sqrt(3)) * CFG["mirror"]["inner_radius"]
CFG["sensor"] = {}
CFG["sensor"]["fov_radius_deg"] = 9.0
CFG["sensor"]["housing_overhead"] = 1.1
CFG["sensor"]["hex_pixel_fov_flat2flat_deg"] = 0.1
CFG["sensor"]["num_paxel_on_diagonal"] = [1, 3, 9]
CFG["light_field_geometry"] = {}
CFG["light_field_geometry"]["num_blocks"] = 24
CFG["light_field_geometry"]["num_photons_per_block"] = 1000 * 1000


def init(work_dir, config=CFG, map_and_reduce_pool=queue_map_reduce):
    os.makedirs(work_dir, exist_ok=True)

    for mkey in merlict.MIRRORS:
        mirror_dir = os.path.join(work_dir, mkey)
        os.makedirs(mirror_dir, exist_ok=True)

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            paxel_dir = os.path.join(mirror_dir, "paxel{:d}".format(npax))
            os.makedirs(paxel_dir, exist_ok=True)

            scenery_dir = os.path.join(paxel_dir, "input", "scenery")
            os.makedirs(scenery_dir, exist_ok=True)
            with open(os.path.join(scenery_dir, "scenery.json"), "wt") as f:
                s = merlict.make_plenoscope_scenery_for_merlict(
                    mirror_key=mkey,
                    num_paxel_on_diagonal=npax,
                    cfg=config,
                )
                f.write(json_numpy.dumps(s))

            lfg_path = os.path.join(paxel_dir, "light_field_geometry")
            if not os.path.exists(lfg_path):
                plenoirf._estimate_light_field_geometry_of_plenoscope(
                    cfg={"light_field_geometry": {
                            "num_blocks": config["light_field_geometry"]["num_blocks"] * npax ** 2,
                            "num_photons_per_block": config["light_field_geometry"]["num_photons_per_block"],
                        }
                    },
                    out_absdir=paxel_dir,
                    map_and_reduce_pool=map_and_reduce_pool,
                    executables=EXECUTABLES,
                )
