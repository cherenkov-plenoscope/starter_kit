#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])

for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )
        traj = {}
        traj[spt.IDX] = event_table["reconstructed_trajectory"][spt.IDX]
        traj["x"] = event_table["reconstructed_trajectory"]["x_m"]
        traj["y"] = event_table["reconstructed_trajectory"]["y_m"]
        traj["cx"] = event_table["reconstructed_trajectory"]["cx_rad"]
        traj["cy"] = event_table["reconstructed_trajectory"]["cy_rad"]
        irf.json_numpy.write(
            path=os.path.join(site_particle_dir, "reco" + ".json"),
            out_dict=traj,
        )
