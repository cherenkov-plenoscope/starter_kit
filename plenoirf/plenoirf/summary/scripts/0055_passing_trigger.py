#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

trigger_modus = sum_config["trigger"]["modus"]
trigger_threshold = sum_config["trigger"]["threshold_pe"]

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:

        site_particle_dir = os.path.join(pa["out_dir"], site_key, particle_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        idx_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=event_table["trigger"],
            threshold=trigger_threshold,
            modus=trigger_modus,
        )

        irf.json_numpy.write(
            path=os.path.join(site_particle_dir, "idx.json"),
            out_dict=idx_pasttrigger,
        )
