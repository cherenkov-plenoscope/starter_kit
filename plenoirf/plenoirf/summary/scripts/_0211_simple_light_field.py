#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])

passing_trigger = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

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

        evt_tab_trg = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=passing_trigger[sk][pk]["passed_trigger"][spt.IDX],
            level_keys=["reconstructed_trajectory"],
        )

        traj = {}
        traj[spt.IDX] = evt_tab_trg["reconstructed_trajectory"][spt.IDX]
        traj["x"] = evt_tab_trg["reconstructed_trajectory"]["x_m"]
        traj["y"] = evt_tab_trg["reconstructed_trajectory"]["y_m"]
        traj["cx"] = evt_tab_trg["reconstructed_trajectory"]["cx_rad"]
        traj["cy"] = evt_tab_trg["reconstructed_trajectory"]["cy_rad"]

        irf.json_numpy.write(
            path=os.path.join(site_particle_dir, "reco" + ".json"),
            out_dict=traj,
        )
